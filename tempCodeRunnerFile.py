    data, cluster_summary, cluster_names = generate_cluster_names(data, 'Cluster', data.select_dtypes(include=['int64', 'float64']).columns[:2])
