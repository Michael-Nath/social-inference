import { DomHighlighter } from './utils/domHighlighter.js';

export class UIManager {
    sessionsContainer;
    currentPartitionElement;
    errorDisplayElement; // Optional: if you have a dedicated error display area

    constructor(containerSelectors) {
        this.sessionsContainer = document.getElementById(containerSelectors.sessionsContainerId);
        this.currentPartitionElement = document.getElementById(containerSelectors.currentPartitionId);
        this.errorDisplayElement = document.getElementById(containerSelectors.errorDisplayId); // Optional

        if (!this.sessionsContainer) {
            console.error("UIManager: Sessions container not found!");
        }
        if (!this.currentPartitionElement) {
            console.error("UIManager: Current partition element not found!");
        }
        DomHighlighter.addHighlightStyles(); // Ensure styles are injected
    }

    displayCurrentPartition(partitionName) {
        if (this.currentPartitionElement) {
            this.currentPartitionElement.textContent = partitionName || 'N/A';
        }
    }

    displayProfiling(profiler) {
        const profilingContainer = document.getElementById('profiling');
        profilingContainer.innerHTML = '';
        
        const totalTime = profiler.getWorkTime();
        const sessionTimes = profiler.getSessionTimes();
        const [cpuKernelTimes, gpuKernelTimes] = profiler.getKernelTimes();
        
        // Total time
        profilingContainer.innerHTML += `<h2>Total Time: ${totalTime.toFixed(3)}ms</h2>`;
        
        // Session breakdown
        profilingContainer.innerHTML += '<h2>Sessions</h2>';
        if (sessionTimes.size === 0) {
            profilingContainer.innerHTML += '<div>No sessions recorded</div>';
        } else {
            // Filter sessions with >1% and sort by time (descending)
            const filteredSessions = Array.from(sessionTimes.entries())
                .filter(([_, time]) => totalTime > 0 && (time / totalTime * 100) > 1)
                .sort((a, b) => b[1] - a[1]);
            
            if (filteredSessions.length === 0) {
                profilingContainer.innerHTML += '<div>No sessions with >1% of total time</div>';
            } else {
                for (const [sessionName, time] of filteredSessions) {
                    const percentage = (time / totalTime * 100).toFixed(3);
                    profilingContainer.innerHTML += `<div>${sessionName}: ${time.toFixed(3)}ms (${percentage}%)</div>`;
                }
            }
        }
        
        // CPU kernels
        profilingContainer.innerHTML += '<h2>CPU Kernels</h2>';
        if (cpuKernelTimes.size === 0) {
            profilingContainer.innerHTML += '<div>No CPU kernels recorded</div>';
        } else {
            // Sort by time (descending)
            const sortedCpuKernels = Array.from(cpuKernelTimes.entries())
                .sort((a, b) => b[1] - a[1]);
            
            for (const [kernelName, time] of sortedCpuKernels) {
                const percentage = totalTime > 0 ? (time / totalTime * 100).toFixed(3) : '0.000';
                profilingContainer.innerHTML += `<div>${kernelName}: ${time.toFixed(3)}ms (${percentage}%)</div>`;
            }
        }
        
        // GPU kernels
        profilingContainer.innerHTML += '<h2>GPU Kernels</h2>';
        if (gpuKernelTimes.size === 0) {
            profilingContainer.innerHTML += '<div>No GPU kernels recorded</div>';
        } else {
            // Sort by time (descending)
            const sortedGpuKernels = Array.from(gpuKernelTimes.entries())
                .sort((a, b) => b[1] - a[1]);
            
            for (const [kernelName, time] of sortedGpuKernels) {
                const percentage = totalTime > 0 ? (time / totalTime * 100).toFixed(3) : '0.000';
                profilingContainer.innerHTML += `<div>${kernelName}: ${time.toFixed(3)}ms (${percentage}%)</div>`;
            }
        }
    }

    displayDecodedText(text) {
        if (this.currentPartitionElement && text && text.trim()) {
            // Create or get the decoded text container
            let decodedTextDiv = document.getElementById('decoded-text-container');
            if (!decodedTextDiv) {
                decodedTextDiv = document.createElement('div');
                decodedTextDiv.id = 'decoded-text-container';
                decodedTextDiv.style.cssText = 'margin-top: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; background: #f9f9f9; font-family: monospace; white-space: pre-wrap;';
                this.currentPartitionElement.parentNode.insertBefore(decodedTextDiv, this.currentPartitionElement.nextSibling);
            }
            decodedTextDiv.textContent = text;
        }
    }

    displayError(message) {
        console.error("UIManager Displaying Error:", message);
        if (this.errorDisplayElement) {
            this.errorDisplayElement.textContent = `Error: ${message || 'Unknown error'}`;
            this.errorDisplayElement.style.color = 'red';
            this.errorDisplayElement.style.display = 'block';
        }
        // Potentially also log to a more prominent UI element if needed
    }

    clearError() {
        if (this.errorDisplayElement) {
            this.errorDisplayElement.textContent = '';
            this.errorDisplayElement.style.display = 'none';
        }
    }

    // Renders the visual representation of the session graph
    renderSessionGraph(sessionGraph) {
        console.log("UIManager: renderSessionGraph called with sessionGraph:", JSON.parse(JSON.stringify(sessionGraph))); // Log the input

        if (!this.sessionsContainer) {
            console.error("UIManager: sessionsContainer is null in renderSessionGraph. Cannot proceed.");
            return;
        }
        this.sessionsContainer.innerHTML = ''; // Clear previous content
        console.log("UIManager: Cleared sessionsContainer.");

        let sessionsToDisplay = [];
        if (sessionGraph && sessionGraph.sessions && Array.isArray(sessionGraph.sessions)) {
            sessionsToDisplay = sessionGraph.sessions;
            console.log("UIManager: sessionsToDisplay initialized with", sessionsToDisplay.length, "sessions.");
        } else {
            console.warn("UIManager: Could not determine sessions to display from sessionGraph structure.", sessionGraph);
            this.sessionsContainer.textContent = 'No valid session graph structure found to display.';
            return;
        }

        if (sessionsToDisplay.length === 0) {
            this.sessionsContainer.textContent = 'No sessions in the graph.';
            console.log("UIManager: No sessions in the graph to display.");
            return;
        }

        sessionsToDisplay.forEach((session, sessionIndex) => {
            console.log("UIManager: Processing session (index " + sessionIndex + "):", JSON.parse(JSON.stringify(session)));

            const sessionIdForHTML = session.id !== undefined ? session.id : sessionIndex;
            console.log(`UIManager: For session index ${sessionIndex}, sessionIdForHTML = '${sessionIdForHTML}' (type: ${typeof sessionIdForHTML})`);

            const sessionDivId = DomHighlighter.getSessionElementId(sessionIdForHTML, sessionIndex);
            console.log(`UIManager: Calculated sessionDivId for session index ${sessionIndex} (using '${sessionIdForHTML}'): '${sessionDivId}'`);

            const sessionDiv = document.createElement('div');
            sessionDiv.className = 'session';
            sessionDiv.id = sessionDivId; // Assign ID here
            sessionDiv.style.border = '1px solid #ccc';
            sessionDiv.style.marginBottom = '10px';
            sessionDiv.style.padding = '10px';
            console.log("UIManager: Created sessionDiv with id:", sessionDiv.id, sessionDiv);

            const sessionTitle = document.createElement('h3');
            sessionTitle.textContent = `Session ID: ${sessionIdForHTML}`; // Using the same determined ID
            sessionTitle.style.marginTop = '0';
            console.log("UIManager: Appending sessionTitle to sessionDiv for session index " + sessionIndex);
            sessionDiv.appendChild(sessionTitle);

            if (session.nodes && Array.isArray(session.nodes) && session.nodes.length > 0) {
                const nodesList = document.createElement('ul');
                nodesList.style.paddingLeft = '20px';
                console.log("UIManager: Created nodesList for session index " + sessionIndex);

                session.nodes.forEach((node, nodeIndexInSession) => {
                    console.log(`UIManager: Processing node (session index ${sessionIndex}, node index ${nodeIndexInSession}):`, JSON.parse(JSON.stringify(node)));

                    const nodeIdentifier = node.name || node.id || `node_${nodeIndexInSession}`;
                    console.log(`UIManager: For node (session index ${sessionIndex}, node index ${nodeIndexInSession}), nodeIdentifier = '${nodeIdentifier}'`);

                    const nodeItemId = DomHighlighter.getNodeElementId(sessionIdForHTML, sessionIndex, nodeIdentifier, nodeIndexInSession);
                    console.log(`UIManager: Calculated nodeItemId for node '${nodeIdentifier}': '${nodeItemId}'`);

                    const nodeItem = document.createElement('li');
                    nodeItem.className = 'node-item';
                    nodeItem.id = nodeItemId; // Assign ID here
                    nodeItem.textContent = `Node: ${nodeIdentifier}`;
                    console.log("UIManager: Created nodeItem with id:", nodeItem.id, nodeItem);

                    // Basic dependency display (can be enhanced)
                    if (node.dependencies && Array.isArray(node.dependencies) && node.dependencies.length > 0) {
                        const dependenciesList = document.createElement('ul');
                        dependenciesList.style.listStyleType = 'circle';
                        dependenciesList.style.marginLeft = '20px';
                        node.dependencies.forEach(dep => {
                            const dependencyItem = document.createElement('li');
                            let depText = 'Unknown Dependency';
                             if (typeof dep === 'string') depText = dep;
                             else if (dep && typeof dep === 'object') depText = dep.node_id || dep.id || JSON.stringify(dep);
                            dependencyItem.textContent = `Depends on: ${depText}`;
                            dependenciesList.appendChild(dependencyItem);
                        });
                        nodeItem.appendChild(dependenciesList);
                    }
                    console.log("UIManager: Appending nodeItem to nodesList for node " + nodeIdentifier);
                    nodesList.appendChild(nodeItem);
                });
                console.log("UIManager: Appending nodesList to sessionDiv for session index " + sessionIndex);
                sessionDiv.appendChild(nodesList);
            } else {
                const noNodesPara = document.createElement('p');
                noNodesPara.textContent = 'No nodes in this session.';
                sessionDiv.appendChild(noNodesPara);
                console.log("UIManager: Added 'No nodes' paragraph for session index " + sessionIndex);
            }
            console.log("UIManager: Appending sessionDiv to sessionsContainer for session index " + sessionIndex, sessionDiv);
            this.sessionsContainer.appendChild(sessionDiv);
        });
        console.log("UIManager: renderSessionGraph completed.");
    }

    // --- Callbacks for SessionExecutor ---
    onSessionStart(sessionId, sessionIndex) {
        return
        const elId = DomHighlighter.getSessionElementId(sessionId, sessionIndex);
        DomHighlighter.updateElementClass(elId, 'session-executing', true);
        DomHighlighter.updateElementClass(elId, 'session-completed', false);
        DomHighlighter.updateElementClass(elId, 'session-failed', false);
    }

    onSessionEnd(sessionId, sessionIndex, success) {
        return
        const elId = DomHighlighter.getSessionElementId(sessionId, sessionIndex);
        DomHighlighter.updateElementClass(elId, 'session-executing', false);
        DomHighlighter.updateElementClass(elId, success ? 'session-completed' : 'session-failed', true);
    }

    onNodeStart(sessionId, sessionIndex, nodeIdentifier, nodeIndexInSession) {
        return
        const elId = DomHighlighter.getNodeElementId(sessionId, sessionIndex, nodeIdentifier, nodeIndexInSession);
        DomHighlighter.updateElementClass(elId, 'node-executing', true);
        DomHighlighter.updateElementClass(elId, 'node-completed', false);
        DomHighlighter.updateElementClass(elId, 'node-failed', false);
    }

    onNodeEnd(sessionId, sessionIndex, nodeIdentifier, nodeIndexInSession, success) {
        console.log("success")
        console.log(success)
        return
        const elId = DomHighlighter.getNodeElementId(sessionId, sessionIndex, nodeIdentifier, nodeIndexInSession);
        DomHighlighter.updateElementClass(elId, 'node-executing', false);
        DomHighlighter.updateElementClass(elId, success ? 'node-completed' : 'node-failed', true);
    }
} 