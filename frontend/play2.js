import { Coordinator } from "./worker.js";

const coordinator = new Coordinator({
    url: "",
});

const registration = await coordinator.register();
console.log(registration);
const work = await coordinator.get_work(registration.partition);
work.graph.dump();

