run:
    go build && ./cvproject /dev/video0 tensorflow_inception_graph.pb imagenet_comp_graph_label_strings.txt
