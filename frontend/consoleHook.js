const hookConsoleMethod = (method, color) => {
    const original = console[method];
    console[method] = (...args) => {
        original(...args);
        
        // Check if any arg is an Error object
        const enhancedArgs = args.map(arg => {
            if (arg instanceof Error) {
                return `${arg.message}\nStack: ${arg.stack}`;
            }
            return arg;
        });
        
        const message = enhancedArgs.join(" ");
        const element = document.getElementById("console");
        element.innerHTML += `<div style="background-color: ${color}">${message}</div>`;
    };
};

hookConsoleMethod("log", "#f0f0f0");    // Light gray
hookConsoleMethod("error", "#ffebee");  // Light red
hookConsoleMethod("debug", "#e8f5e9");  // Light green
hookConsoleMethod("info", "#e3f2fd");   // Light blue
hookConsoleMethod("warn", "#fff3e0");   // Light orange