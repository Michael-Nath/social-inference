import { CPUKernel, GPUKernel } from "../kernel.js";

export class Profiler {
    constructor() {
        this.nextTicketId = 0;
        this.activeWork = new Map(); // ticketId -> startTime
        this.activeSessions = new Map(); // ticketId -> { name, startTime }
        this.activeKernels = new Map(); // ticketId -> { kernel, startTime }
        
        this.sessionTimes = new Map(); // sessionName -> cumulativeTime
        this.cpuKernelTimes = new Map(); // kernelName -> cumulativeTime
        this.gpuKernelTimes = new Map(); // kernelName -> cumulativeTime
        this.totalWorkTime = 0;
    }

    enterWork() {
        const ticket = this.nextTicketId++;
        this.activeWork.set(ticket, performance.now());
        return ticket;
    }

    enterSession(session) {
        const ticket = this.nextTicketId++;
        this.activeSessions.set(ticket, {
            name: session.index,
            startTime: performance.now()
        });
        return ticket;
    }

    enterKernel(kernel) {
        const ticket = this.nextTicketId++;
        this.activeKernels.set(ticket, {
            kernel: kernel,
            startTime: performance.now()
        });
        return ticket;
    }

    exitKernel(ticket) {
        if (!this.activeKernels.has(ticket)) {
            throw new Error(`Invalid kernel ticket: ${ticket}`);
        }
        
        const { kernel, startTime } = this.activeKernels.get(ticket);
        const duration = performance.now() - startTime;
        
        if (kernel instanceof CPUKernel) {
            const currentTime = this.cpuKernelTimes.get(kernel.name) || 0;
            this.cpuKernelTimes.set(kernel.name, currentTime + duration);
        } else if (kernel instanceof GPUKernel) {
            const currentTime = this.gpuKernelTimes.get(kernel.name) || 0;
            this.gpuKernelTimes.set(kernel.name, currentTime + duration);
        }
        
        this.activeKernels.delete(ticket);
    }

    exitSession(ticket) {
        if (!this.activeSessions.has(ticket)) {
            throw new Error(`Invalid session ticket: ${ticket}`);
        }
        
        const { name, startTime } = this.activeSessions.get(ticket);
        const duration = performance.now() - startTime;
        
        const currentTime = this.sessionTimes.get(name) || 0;
        this.sessionTimes.set(name, currentTime + duration);
        
        this.activeSessions.delete(ticket);
    }

    exitWork(ticket) {
        if (!this.activeWork.has(ticket)) {
            throw new Error(`Invalid work ticket: ${ticket}`);
        }
        
        const startTime = this.activeWork.get(ticket);
        const duration = performance.now() - startTime;
        this.totalWorkTime += duration;
        
        this.activeWork.delete(ticket);
    }

    getSessionTimes() {
        return new Map(this.sessionTimes);
    }

    getKernelTimes() {
        return [
            new Map(this.cpuKernelTimes),
            new Map(this.gpuKernelTimes)
        ];
    }

    getWorkTime() {
        return this.totalWorkTime;
    }
}