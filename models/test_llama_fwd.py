import torch
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRMSNorm, LlamaDecoderLayer
from transformers.models.llama.configuration_llama import LlamaConfig
from .llama import llama_attn, layernorm, llama_fwd
from .utils import load_model
from inference.tensor import Tensor
from inference.test_util import llama_1b_cache
from inference.simulator import simulate
from inference.graph import ComputeGraphBuilder
from inference.pipeline import ComputePipeline, PipelineInput


def package_llama_decoder_layer_weights(layer: LlamaDecoderLayer, b: ComputeGraphBuilder) -> dict:
    """
    Extracts weights from a PyTorch LlamaDecoderLayer and packages them as graph nodes.
    """
    packaged_weights = {
        "input_layernorm": {},
        "self_attn": {},
        "mlp": {},
        "post_attention_layernorm": {}
    }
    all_param_nodes = {} # Temporary dict to hold nodes by their original HF names

    # Create graph nodes for all learnable parameters
    for name, param in layer.named_parameters():
        graph_node_name = f"weights/{name.replace('.', '/')}"
        data = param.data.clone()
        if len(data.shape) == 2:
            data = data.T
        node = b.fixed(graph_node_name, data.detach())
        all_param_nodes[name] = node

    # Handle input layernorm
    if "input_layernorm.weight" in all_param_nodes:
        packaged_weights["input_layernorm"]["weight"] = all_param_nodes["input_layernorm.weight"]
    ln_eps_tensor = torch.tensor(layer.input_layernorm.variance_epsilon, dtype=torch.float32)
    packaged_weights["input_layernorm"]["eps"] = b.fixed("params/input_layernorm.eps", ln_eps_tensor.unsqueeze(0))


    # Handle self-attention weights
    # Mapping from graph function expected keys to Hugging Face parameter names
    attn_key_map = {
        "q_weight": "self_attn.q_proj.weight",
        "k_weight": "self_attn.k_proj.weight",
        "v_weight": "self_attn.v_proj.weight",
        "o_weight": "self_attn.o_proj.weight",
    }
    for graph_key, hf_key in attn_key_map.items():
        if hf_key in all_param_nodes:
            packaged_weights["self_attn"][graph_key] = all_param_nodes[hf_key]

    # Handle MLP weights
    mlp_key_map = {
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj": "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight",
    }
    for graph_key, hf_key in mlp_key_map.items():
        if hf_key in all_param_nodes:
            packaged_weights["mlp"][graph_key] = all_param_nodes[hf_key]

    # Handle post-attention layernorm
    if "post_attention_layernorm.weight" in all_param_nodes:
        packaged_weights["post_attention_layernorm"]["weight"] = all_param_nodes["post_attention_layernorm.weight"]
    post_ln_eps_tensor = torch.tensor(layer.post_attention_layernorm.variance_epsilon, dtype=torch.float32)
    packaged_weights["post_attention_layernorm"]["eps"] = b.fixed("params/post_attention_layernorm.eps", post_ln_eps_tensor.unsqueeze(0))

    return packaged_weights


def test_llama_layernorm():
    # fetch the weights from a layernorm module of llama 
    batch_size = 1
    seq_len = 1
    hidden_dim = 2048
    rms_norm_eps = 1e-5
    layernorm_module = LlamaRMSNorm(hidden_size=hidden_dim, eps=rms_norm_eps)

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # hardcode the weights
    layer_weights = torch.randn(hidden_dim)
    layernorm_module.weight.data = layer_weights.clone()

    with torch.no_grad():
        gt_output = layernorm_module(hidden_states)
        
    builder = ComputeGraphBuilder()
    # Create input nodes
    hidden_states_node = builder.input("hidden_states")
    eps_node = builder.input("eps")
    weight_node = builder.input("weights")

    with builder.partition("p0"):
        layernorm_node = layernorm(builder, hidden_states_node, weight_node, eps_node)
    builder.output("layernorm_output", layernorm_node)

    # Build graph
    g = builder.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
        "eps": Tensor.from_torch(torch.tensor([rms_norm_eps])),
        "weights": Tensor.from_torch(layer_weights)
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))

    # Get partition work
    work = pipeline.get_partition_work("p0")
    assert work is not None
    
    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)
    our_output = result.outputs[0].tensor.to_torch()
    assert torch.allclose(gt_output, our_output, rtol=1e-3)

        

def test_llama_attn():
    # Test parameters
    batch_size = 1
    num_heads = 1
    seq_len = 1
    head_dim = 64
    hidden_dim = num_heads * head_dim

    # Input
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    # Hardcoded weights
    mock_q_weight = torch.randn(hidden_dim, hidden_dim)
    mock_k_weight = torch.randn(hidden_dim, hidden_dim)
    mock_v_weight = torch.randn(hidden_dim, hidden_dim)
    mock_o_weight = torch.randn(hidden_dim, hidden_dim)

    # Create a real LlamaAttention object
    config = LlamaConfig(hidden_size=hidden_dim, num_attention_heads=num_heads)
    attn = LlamaAttention(config, 0)

    # Replace projection weights with hardcoded ones
    attn.q_proj.weight.data = mock_q_weight.clone()
    attn.k_proj.weight.data = mock_k_weight.clone()
    attn.v_proj.weight.data = mock_v_weight.clone()
    attn.o_proj.weight.data = mock_o_weight.clone()

    # Forward pass
    with torch.no_grad():
        gt_output, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=(cos, sin),
            attention_mask=None
        )

    # Run our implementation
    builder = ComputeGraphBuilder()
    
    # Create input nodes
    hidden_states_node = builder.input("hidden_states")
    q_weight = builder.input("q_weight")
    k_weight = builder.input("k_weight")
    v_weight = builder.input("v_weight")
    o_weight = builder.input("o_weight")
    cos_node = builder.input("cos")
    sin_node = builder.input("sin")
    
    # Apply RoPE
    with builder.partition("p0"):
        attn_output = llama_attn(builder,
            hidden_states_node, head_dim, q_weight=q_weight, k_weight=k_weight, v_weight=v_weight, o_weight=o_weight, n_kv_heads=num_heads, position_embeddings=(cos_node, sin_node)
        )
    
    # Create output nodes
    builder.output("attn_output", attn_output)
    
    # Build graph
    g = builder.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
        "q_weight": Tensor.from_torch(mock_q_weight.T),
        "k_weight": Tensor.from_torch(mock_k_weight.T),
        "v_weight": Tensor.from_torch(mock_v_weight.T),
        "o_weight": Tensor.from_torch(mock_o_weight.T),
        "cos": Tensor.from_torch(cos),
        "sin": Tensor.from_torch(sin)
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))

    # Get partition work
    work = pipeline.get_partition_work("p0")
    assert work is not None
    
    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)
    our_attn_output = result.outputs[0].tensor.to_torch()
    assert torch.allclose(gt_output, our_attn_output, rtol=1e-3)



def test_llama_fwd():
    
    model_path = "meta-llama/Llama-3.2-1B"
    # Load the model
    model = load_model(model_path)
    model.model = model.model.float()
    

    batch_size = 1
    seq_len = 1
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    
    
    head_dim = model.config.head_dim
    hidden_dim = num_heads * head_dim
    # create some hidden states
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)
    
    # Extract the first decoder layer
    first_decoder_layer: LlamaDecoderLayer = model.model.layers[0].cpu()

    b = ComputeGraphBuilder()
    with b.partition("p0"):
    
        # Use the new utility function to package weights
        packaged_layer_weights = package_llama_decoder_layer_weights(first_decoder_layer, b)
        
        # Prepare the specific weight_dict for the current llama_fwd function
        weight_dict_for_llama_fwd = {
            "input_layernorm": packaged_layer_weights["input_layernorm"],
            "attn": packaged_layer_weights["self_attn"],
            # As llama_fwd expands, you'll add more here, e.g.:
            "mlp": packaged_layer_weights["mlp"],
            "post_layernorm": packaged_layer_weights["post_attention_layernorm"]
        }

        # Create a graph node for the hidden_states input tensor
        hidden_states_node = b.input("hidden_states")

        # Now you can call your graph implementation of llama_fwd
        cos_node = b.fixed("cos", cos)
        sin_node = b.fixed("sin", sin)
        fwd_output_node = llama_fwd(
            b, hidden_states_node, head_dim, num_kv_heads, model.config.hidden_act, weight_dict_for_llama_fwd,
            (cos_node, sin_node) 
        )
    b.output("llama_fwd_out", fwd_output_node)

    # Build graph
    g = b.build()

    # Create pipeline and enqueue inputs
    pipeline = ComputePipeline(g)
    inputs = {
        "hidden_states": Tensor.from_torch(hidden_states),
    }
    pipeline.enqueue_input(PipelineInput(correlation_id="test", inputs=inputs))
    work = pipeline.get_partition_work("p0")
    assert work is not None
    
    # Run simulation
    cache = llama_1b_cache()
    result = simulate(work, cache)
    
    our_output = None
    for output in result.outputs:
        if output.node == fwd_output_node.name:
            our_output = output.tensor.to_torch()
            break
    gt_output = first_decoder_layer(hidden_states, position_embeddings = (cos, sin))[0]

    assert torch.allclose(our_output, gt_output, rtol=1e-3)
    
def test_llama_model():
    

if __name__ == "__main__":
    test_llama_attn()