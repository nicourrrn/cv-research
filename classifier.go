package main

import (
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

// processImage processes a single image frame for classification
func processImage(img gocv.Mat, net gocv.Net, descriptions []string, statusColor color.RGBA) string {
	// Convert image Mat to 224x224 blob that the classifier can analyze
	blob := gocv.BlobFromImage(img, 1.0, image.Pt(224, 224), gocv.NewScalar(0, 0, 0, 0), true, false)

	// Feed the blob into the classifier
	net.SetInput(blob, "input")

	// Run a forward pass thru the network
	prob := net.Forward("softmax2")

	// Reshape the results into a 1x1000 matrix
	probMat := prob.Reshape(1, 1)

	// Determine the most probable classification
	_, maxVal, _, maxLoc := gocv.MinMaxLoc(probMat)

	// Display classification
	desc := "Unknown"
	if maxLoc.X < len(descriptions) {
		desc = descriptions[maxLoc.X]
	}
	status := fmt.Sprintf("description: %v, maxVal: %v", desc, maxVal)

	gocv.PutText(&img, status, image.Pt(10, 20), gocv.FontHersheyPlain, 1.2, statusColor, 2)

	blob.Close()
	prob.Close()
	probMat.Close()

	return status
}
