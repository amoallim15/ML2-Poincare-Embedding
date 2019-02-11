# ML2-Poincare-Embedding

This repository is a simple implementation of [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) paper introduced by Maximilian Nickel and Douwe Kiela at Facebook AI Research.

## Summary

This paper introduced an interesting model to learn vector representation of words in a graph.
It takes a list of relations between words such as [ [banana fruit], [eatable_fruit fruit] ], and attempts to learn its vector representation such that the distance between the words' vectors accurately represent how close the words are in the graph.
The navality of this paper is by introducing a new approach to model hierarchical structures, as opposed to the commonly used Euclidean space, as it embeds words into hyperbolic space, or more precisely into an n-dimentional Poincare ball.
The reason presented for this is that hyperbolic spaces are more suitable for capturing hierarchical and similarity information of the words and inherently present it in the graph.

### Distance Function

The model calculates the distances between two words' vectors using the following fuction:
Where:
	u, v are multi-dimentional vectors of any two words in the dataset.

The distances within the Poincare ball changes smoothly with respect to the location of the u and v vectors.
This locality property of the Poincare distance is key for finding continous 

### Loss Function

The model achives this by defining a loss function that minimizes the distance between connected words and maximizing the distances between unconnected words.
and then training over the list of connected nodes, along with randomly sampled negative examples, using gradient descent.

