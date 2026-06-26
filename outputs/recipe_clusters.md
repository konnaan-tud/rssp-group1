# Recipe Clusters

Similarity is the symmetric hybrid metric used by the belief updater (F1 + order bonus), normalised to the **[0, 1]** range by dividing by (1 + alpha). Cluster count was chosen automatically by silhouette score (best k = 2, silhouette = 0.338).

## Per-k silhouette scores

- k = 2: 0.338 ←
- k = 3: 0.282
- k = 4: 0.252
- k = 5: 0.222
- k = 6: 0.202
- k = 7: 0.233
- k = 8: 0.195

## Cluster 1 — 27 recipes (avg internal similarity 0.730)

- alfredo pasta
- amatriciana pasta
- arrabbiata pasta
- broccoli garlic pasta
- butter parmesan pasta
- cacio e pepe
- cacio e uova
- carbonara
- carbonara bianca
- clam pasta
- garlic oil pasta
- gnocchi pomodoro
- gricia
- ham and pea pasta
- lemon butter pasta
- marinara pasta
- mushroom cream pasta
- pea butter pasta
- pea pesto pasta
- pomodoro pasta
- puttanesca pasta
- scrambled carbonara
- shrimp garlic pasta
- spinach pesto pasta
- tuna tomato pasta
- vodka sauce pasta
- zucchini sauce pasta

## Cluster 2 — 2 recipes (avg internal similarity 0.732)

- gnocchi pesto
- pesto pasta
