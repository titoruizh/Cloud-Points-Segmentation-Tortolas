"""
Estilos CSS Personalizados
===========================
Tema moderno y elegante para la aplicación.
"""


def get_custom_css() -> str:
    """
    Retorna el CSS personalizado para la aplicación.
    
    Returns:
        String con estilos CSS
    """
    return """
    /* ===== Tema Principal ===== */
    :root {
        --primary-color: #6366f1;
        --primary-hover: #4f46e5;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --bg-dark: #1e1e2e;
        --bg-card: #2d2d3d;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border-color: #3d3d5c;
    }
    
    /* ===== Header ===== */
    .app-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    
    .app-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .app-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-top: 8px;
    }
    
    /* ===== Cards ===== */
    .config-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .config-card h3 {
        color: var(--text-primary);
        font-size: 1.1rem;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* ===== Status Badges ===== */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(16, 185, 129, 0.2);
        color: var(--success-color);
        border: 1px solid var(--success-color);
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: var(--warning-color);
        border: 1px solid var(--warning-color);
    }
    
    .status-error {
        background: rgba(239, 68, 68, 0.2);
        color: var(--error-color);
        border: 1px solid var(--error-color);
    }
    
    /* ===== Progress Log ===== */
    .progress-log {
        background: #1a1a2e;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.9rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .progress-log p {
        margin: 4px 0;
        line-height: 1.6;
    }
    
    /* ===== Buttons ===== */
    .primary-btn {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    
    .primary-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    /* ===== File Upload ===== */
    .file-upload-area {
        border: 2px dashed var(--border-color);
        border-radius: 12px;
        padding: 40px;
        text-align: center;
        transition: all 0.3s ease;
        background: rgba(99, 102, 241, 0.05);
    }
    
    .file-upload-area:hover {
        border-color: var(--primary-color);
        background: rgba(99, 102, 241, 0.1);
    }
    
    /* ===== Stats Cards ===== */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    
    .stat-card .value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stat-card .label {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    /* ===== Pipeline Steps ===== */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        background: var(--bg-card);
        border-left: 3px solid var(--border-color);
    }
    
    .pipeline-step.active {
        border-left-color: var(--primary-color);
        background: rgba(99, 102, 241, 0.1);
    }
    
    .pipeline-step.completed {
        border-left-color: var(--success-color);
    }
    
    .pipeline-step.error {
        border-left-color: var(--error-color);
    }
    
    /* ===== Tooltips ===== */
    .tooltip {
        position: relative;
    }
    
    .tooltip::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: var(--bg-dark);
        color: var(--text-primary);
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        white-space: nowrap;
        opacity: 0;
        visibility: hidden;
        transition: all 0.2s ease;
    }
    
    .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }
    
    /* ===== Scrollbar ===== */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* ===== Animations ===== */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing {
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .slide-in {
        animation: slideIn 0.3s ease-out;
    }
    """
