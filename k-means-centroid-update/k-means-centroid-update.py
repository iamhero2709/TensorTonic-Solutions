def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    d = len(points[0])   # dimension
    
    # Step 1: initialize sums and counts
    sums = [[0.0] * d for _ in range(k)]
    counts = [0] * k
    
    # Step 2: accumulate sums
    for point, cluster_id in zip(points, assignments):
        for i in range(d):
            sums[cluster_id][i] += point[i]
        counts[cluster_id] += 1
    
    # Step 3: compute centroids
    centroids = []
    for j in range(k):
        if counts[j] == 0:
            centroids.append([0.0] * d)
        else:
            centroids.append([sums[j][i] / counts[j] for i in range(d)])
    
    return centroids