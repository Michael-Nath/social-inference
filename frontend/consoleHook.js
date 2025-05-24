const hookConsoleMethod = (method, color) => {
    const original = console[method];
    console[method] = (...args) => {
        original(...args);
        const message = args.join(" ");
        const element = document.getElementById("console");
        element.innerHTML += `<div style="background-color: ${color}">${message}</div>`;
    };
};

hookConsoleMethod("log", "#f0f0f0");    // Light gray
hookConsoleMethod("error", "#ffebee");  // Light red
hookConsoleMethod("debug", "#e8f5e9");  // Light green
hookConsoleMethod("info", "#e3f2fd");   // Light blue
hookConsoleMethod("warn", "#fff3e0");   // Light orange