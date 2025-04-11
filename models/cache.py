import threading
import struct
import requests
from dataclasses import dataclass
from contextlib import contextmanager
import os
import torch
from copy import deepcopy
import time
from dotenv import load_dotenv

load_dotenv()

GB = 1024 * 1024 * 1024
SIZE_LIMIT = 10 * GB


def _get_range(url: str, start: int, length: int) -> requests.Response:
  return requests.get(url, headers={
    'Authorization': "Bearer " + os.getenv("HF_TOKEN"),
    'Range': f'bytes={start}-{start+length-1}'
  })

@dataclass
class CacheStatistics:
  """
  Statistics about the cache
  """
  hits: int
  hits_bytes: int
  misses: int
  misses_bytes: int
  evictions: int
  evictions_bytes: int

  present: int
  present_bytes: int

@dataclass
class CacheEntry:
  file: str
  """
  Backing SafeTensor URL
  """

  tensor: str
  """
  SafeTensor name of tensor
  """
 
  last_use: float
  begin: int
  end: int
  dtype: str
  shape: list[int]

  data: torch.Tensor | None

  refcount: threading.Semaphore

  def pinned(self) -> bool:
    return self.refcount.locked()

  def size(self) -> int:
    return self.end - self.begin

  def present(self) -> bool:
    return self.data is not None

  def pin(self) -> bool:
    self.refcount.acquire()

  def unpin(self) -> bool:
    self.refcount.release()

  def fill(self):
    """
    Fill the cache entry
    """
    data = _get_range(self.file, self.begin, self.end - self.begin)
    # Convert the bytes data to a numpy array of the correct shape and dtype
    if data.status_code == 200 or data.status_code == 206:
      # Map SafeTensor dtype to numpy dtype
      dtype_map = {
        "F32": torch.float32,
        "F16": torch.float16,
        "BF16": torch.bfloat16, 
        "I32": torch.int32,
        "I16": torch.int16,
        "I8": torch.int8,
        "U8": torch.uint8,
        "BOOL": torch.bool
      }
      
      if self.dtype.upper() not in dtype_map:
        raise ValueError(f"Unsupported dtype: {self.dtype}")

      torch_dtype = dtype_map[self.dtype.upper()]
      
      # Convert raw bytes to numpy array
      # Convert to bytes to ensure it's writable
      content_bytes = bytes(data.content)
      raw_array = torch.frombuffer(content_bytes, dtype=torch_dtype)
      
      # Reshape to the tensor's shape
      self.data = raw_array.reshape(self.shape)
      self.last_use = time.time()
    else:
      # Handle error case
      raise ValueError(f"Failed to fetch tensor data: HTTP {data.status_code}")

class SafeTensorCache:
  """
  General-purpose SafeTensor cache.
  """  

  files: list[str]
  """
  List of backing SafeTensor URLs
  """

  _cache: dict[str, CacheEntry]
  _lock: threading.Lock
  _stats: CacheStatistics

  def __init__(self, files: list[str]):
    self.files = files
    self._cache = {}
    self._lock = threading.Lock()
    self._stats = CacheStatistics(
      hits=0,
      hits_bytes=0,
      misses=0,
      misses_bytes=0,
      evictions=0,
      evictions_bytes=0,
      present=0,
      present_bytes=0,
    )
    self._build_cache()

  def _build_cache(self, force: bool = False):
    """
    Builds backing dictionary if not already present
    """
    if force:
      if self._cache:
        del self._cache
    else:
      if self._cache:
        return

    for url in self.files:
      # Get header
      r = _get_range(url, 0, 8)
      if len(r.content) != 8:
          raise ValueError(f"Expected 8 bytes for header length, got {len(r.content)}")
      length_of_header = struct.unpack('<Q', r.content)[0]
      header = _get_range(url, 8, length_of_header).json()
      # Parse header
      for k, v in header.items():
        if k == "__metadata__":
          continue
        print(v)
        self._cache[k] = CacheEntry(
          file=url,
          tensor=k,
          begin=v["data_offsets"][0],
          end=v["data_offsets"][1],
          dtype=v["dtype"],
          shape=v["shape"],
          data=None,
          refcount=threading.Semaphore(value=0),
          last_use=0.0
        )

  def get_stats(self) -> CacheStatistics:
    """
    Get current cache statistics
    """
    with self._lock:
      return deepcopy(self._stats)

  def reset_stats(self):
    """
    Reset all cache statistics to zero
    """
    with self._lock:
      self._stats = CacheStatistics(
      hits=0,
      hits_bytes=0,
      misses=0,
      misses_bytes=0,
      evictions=0,
      evictions_bytes=0,
      present=0,
      present_bytes=0,
      )

  @contextmanager
  def get_tensor(self, tensor: str):
    try:
      print("Acquiring lock")
      self._lock.acquire()
      try:
        self._build_cache()

        entry = self._cache[tensor]
        if not entry.present():
          # Cache miss
          self._stats.misses += 1
          self._stats.misses_bytes += entry.size()
          print("Cache miss for tensor", tensor)

          # Evict tensors if needed to make space
          required_space = entry.size()
          current_size = sum(e.size() for e in self._cache.values() if e.present() and not e.pinned())
          
          if current_size + required_space > SIZE_LIMIT:
              # Need to evict some entries
              # Sort unpinned entries by size (smallest first for simplicity)
              evictable = [e for e in self._cache.values() if e.present() and not e.pinned()]
              evictable.sort(key=lambda e: e.last_use)
              
              space_to_free = (current_size + required_space) - SIZE_LIMIT
              freed_space = 0
              
              for e in evictable:
                  if freed_space >= space_to_free:
                      break
                  
                  # Evict this entry
                  e.data = None
                  freed_space += e.size()
                  
                  # Update eviction stats
                  self._stats.evictions += 1
                  self._stats.evictions_bytes += e.size()
                  
                  # Update present stats
                  self._stats.present -= 1
                  self._stats.present_bytes -= e.size()
          entry.fill()
          print("Cache filled for tensor", tensor)
        else:
          # Cache hit
          self._stats.hits += 1
          self._stats.hits_bytes += entry.size()
          entry.last_use = time.time()
          print("Cache hit for tensor", tensor)
        
        # Update present
        if entry.present():
          self._stats.present += 1
          self._stats.present_bytes += entry.size()

        #entry.pin()
      finally:
        self._lock.release()
        print("Released lock")
      yield entry.data
    finally:
      pass
      #entry.unpin()

class ModelCache:
  _caches: dict[str, SafeTensorCache]

  def __init__(self):
    self._caches = {
      "meta-llama/Llama-3.2-1B": SafeTensorCache([
        "https://huggingface.co/meta-llama/Llama-3.2-1B/resolve/main/model.safetensors"
      ])
    }
  def get_cache(self, model: str) -> SafeTensorCache:
    return self._caches[model]