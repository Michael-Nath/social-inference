// frontend/utils/domHighlighter.js
export class DomHighlighter {
    static getSessionElementId(sessionId, sessionIndex) {
        const idPart = sessionId !== undefined && sessionId !== null ? String(sessionId).replace(/[\/\s]/g, '_') : `session_${sessionIndex}`;
        return `session-div-${idPart}`;
    }

    static getNodeElementId(sessionId, sessionIndex, nodeIdentifierInSession, nodeIndexInSession) {
        const sessionPart = sessionId !== undefined && sessionId !== null ? String(sessionId).replace(/[\/\s]/g, '_') : `session_${sessionIndex}`;
        const nodePart = String(nodeIdentifierInSession || `node_${nodeIndexInSession}`).replace(/[\/\s]/g, '_');
        return `node-li-${sessionPart}-${nodePart}`;
    }

    static updateElementClass(elementId, className, add = true) {
        const element = document.getElementById(elementId);
        if (element) {
            if (add) {
                element.classList.add(className);
            } else {
                element.classList.remove(className);
            }
        } else {
            // console.warn(`DomHighlighter: Element with ID '${elementId}' not found for class update.`);
        }
    }

    static addHighlightStyles() {
        const styleId = 'execution-highlight-styles';
        if (document.getElementById(styleId)) return;

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .session-executing {
                background-color: #fff3cd !important;
                border-left: 5px solid #ffc107 !important;
            }
            .node-executing {
                background-color: #d1ecf1 !important;
                font-weight: bold !important;
            }
            .session-completed {
                background-color: #d4edda !important;
                border-left: 5px solid #28a745 !important;
            }
            .node-completed {
                /* background-color: #e2e3e5 !important; */
            }
            .session-failed {
                background-color: #f8d7da !important;
                border-left: 5px solid #dc3545 !important;
            }
            .node-failed {
                color: #721c24 !important;
                background-color: #f5c6cb !important;
                text-decoration: line-through;
            }
            .session, .node-item {
                transition: background-color 0.2s ease-in-out, border-left 0.2s ease-in-out;
            }
        `;
        document.head.appendChild(style);
    }
} 