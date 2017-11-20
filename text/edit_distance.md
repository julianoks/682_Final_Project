# Notes


### (This metric essentially found the permutation that maximally aligns two netwoks, and quantifies how far that permutation is from the identity permutation.) ~~Distance metric #1 (not as feasible, and likely not as well received)~~
1) ~~find a (near) optimal permutation of filters, for each layer.  If we have N layers, we will have to find (N-1) permutations.  This can be seen as solving (N-1) sequence alignment problems, where the elements in each sequence are tensors and the objective is to minimize the sum of distances between all pairs of tensors (by some choice of norm).~~
2) ~~Find the hamming distance (or some other edit distance, eg translocation) between each of the (N-1) permutations and the (N-1) identity permutations.~~

### Distance metric #2
Calculate the number of filters whose nearest neighbor, in the other network, is in the same position. _Note: This is a pairwise operation, and for a layer with N filters, we would need to calculate N choose 2 distances, which is O(N^2).  In our MNIST model , the max N is 512=2^9, so this is not too too bad._

### Distance metric #3
Average relative difference between corresponding parameters.  This is a much simpler and faster calculation, and conveniently the ReLu scaling is divided away.  Again, the relative error between two values, (a,b), is abs((a-b)/a).  If we scale both a and b by λ, we get λ(a-b)/λa = (a-b)/a, so this measure is invariant to the slope, λ.  There is something I didn't mention about relative difference, and that is the denominator, in this case a, is the actual, and b is the observed.  But no one network is the ground truth, so we must make this calculation symmetric by adding the relative differences where each value is the ground truth. This works out to be abs((a+b)(a-b)/ab).  I have to make sure the "two-sided relative difference" makes sense.

##### It would be useful if we could make more polynomial time metrics.  It would also be nice if our results are significant across multiple metrics.


## Questions
1) Related to distance metric #1: Are these sequence alignment problems actually independent?  Consider the case where the first layer's filters are very different from the other network's first layer's.  This will change the ordering of the second layer's weights, which may change the optimal ordering of the second layer's filters.  That would be the base case, and it would propogate to the further layers.  This suggests that they are dependent only on the previous layers, and thus we should solve the permutations in a forwards manner, and correct the next layer.  Though I don't think correcting the layers would make for a very useful or interesting metric - but let's wait and see what the results show us.
