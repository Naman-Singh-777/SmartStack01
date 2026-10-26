# Reducing cloud computing cost

Running your own models on cloud GPUs is billed by the hour, so idle time is wasted money. The biggest savings usually come from a few habits.

Spot instances, called preemptible or low priority machines by some providers, use spare capacity at a steep discount. The provider can take the machine back with little warning, so they suit work that can be retried, such as batch jobs, and they are a poor fit for a request a person is waiting on.

Batching groups several requests together so one GPU handles them at once. Throughput goes up and cost per request goes down, at the price of a short wait. Work that nobody is waiting for is ideal to batch.

Region choice matters. The same machine can cost noticeably more in one region than another. Distance adds network delay, so latency sensitive traffic should stay close to users while background jobs can run wherever it is cheapest.

Right size the hardware. A small model fits on a small GPU, and paying for a top tier GPU to run it wastes money. Match the GPU class to the model.

Autoscaling adds machines when traffic rises and removes them when it falls. Measure utilisation first, because a fleet that sits half empty is the most common source of waste.
