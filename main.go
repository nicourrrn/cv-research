// go run . 0 tensorflow_inception_graph.pb imagenet_comp_graph_label_strings.txt opencv cpu
package main

import (
	"fmt"
	"image/color"
	"os"
	"sync"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 4 {
		panic("How to run:\ntf-classifier [camera ID] [modelfile] [descriptionsfile]")
	}

	// Parse args
	deviceID := os.Args[1]
	model := os.Args[2]
	descr := os.Args[3]
	descriptions, err := readDescriptions(descr)
	if err != nil {
		panic(fmt.Sprintf("Error reading descriptions file: %v\n", err))
	}

	backend := gocv.NetBackendDefault
	if len(os.Args) > 4 {
		backend = gocv.ParseNetBackend(os.Args[4])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 5 {
		target = gocv.ParseNetTarget(os.Args[5])
	}

	// Open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		panic(fmt.Sprintf("Error opening video capture device: %v\n", err))
	}
	defer webcam.Close()

	window := gocv.NewWindow("Tensorflow Classifier")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// Open DNN classifier
	net := gocv.ReadNet(model, "")
	if net.Empty() {
		panic(fmt.Sprintf("Error reading network model: %v\n", model))
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	statusColor := color.RGBA{0, 255, 0, 0}
	fmt.Printf("Start reading device: %v\n", deviceID)

	var wg sync.WaitGroup
	statusChan := make(chan string) // Channel for status updates

	go func() {
		for {
			select {
			case status := <-statusChan:
				fmt.Println(status)
			}
		}
	}()

	for {
		if ok := webcam.Read(&img); !ok {
			panic(fmt.Sprintf("Device closed: %v\n", deviceID))
		}
		if img.Empty() {
			continue
		}

		wg.Add(1)
		go func(img gocv.Mat) {
			defer wg.Done()
			// Process the image in a goroutine and send status back to the main loop
			status := processImage(img, net, descriptions, statusColor)
			statusChan <- status // Send the status to the main loop
		}(img)

		wg.Wait() // Wait for the processing to finish

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}

	wg.Wait() // Wait for any remaining processing before exiting
}
