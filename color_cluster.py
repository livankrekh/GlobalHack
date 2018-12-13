from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import threading
import argparse
import cv2

def get_percent(clt):
	labels = np.unique(clt.labels_)
	imsize = len(clt.labels_)
	res = np.arange(0, len(labels))

	for point in clt.labels_:
		res[point] += 1

	res = res / imsize
	res = np.sort(res)

	print(res)

	if (len(res) == 3):
		if (res[2] >= 0.5 and res[1] - res[0] <= 0.1):
			return True

	if (res[len(res) - 1] - res[0] <= 0.1):
		return True

	return False


def centroid_histogram(clt):
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	hist = hist.astype("float")
	hist /= hist.sum()
 
	return hist

def colors_img(hist, centroids):

	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	return bar

def plot_image(image, n_clusters):

	image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image_list = image_rgb.reshape((image.shape[0] * image.shape[1], 3))
	km_model = KMeans(n_clusters=n_clusters)
	km_model.fit(image_list)

	hist = centroid_histogram(km_model)
	bar = colors_img(hist, km_model.cluster_centers_)
	bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

	image[0:bar.shape[0], 0:bar.shape[1]] = bar

	cv2.imshow("LOL", image)
	cv2.waitKey(0)

def train_model(model, image):
	img_list = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img_list = img_list.reshape((img_list.shape[0] * img_list.shape[1], 3))
	model.fit(img_list)

if __name__ == "__main__":

	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required = False, help = "Path to the image")
	ap.add_argument("-v", "--video", required = False, help = "Path to video")
	ap.add_argument("-c", "--clusters", required = True, type = int,
		help = "# of clusters")
	args = vars(ap.parse_args())

	n_clusters = args["clusters"]

	if (args["image"] == None and args["video"] == None):
		print("Need -i path/to/image or -v path/to/video")

	if (args["image"] != None):
		image = cv2.imread(args["image"])
		plot_image(image, n_clusters)

	cap = cv2.VideoCapture(args["video"])

	model = KMeans(n_clusters=n_clusters)
	status, first_img = cap.read()

	thread_model = threading.Thread(target=train_model, name="train_model", args=[model, first_img])
	thread_model.start()
	capture = first_img
	bar = []
	take_foto = False

	while (status):

		if (thread_model.isAlive() and bar == []):
			cv2.putText(capture, "Please, wait!", (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))
			cv2.imshow("Color analizator", capture)
		elif (thread_model.isAlive() and bar != []):
			hist = centroid_histogram(model)
			bar = colors_img(hist, model.cluster_centers_)
			bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

			if (take_foto):
				cv2.putText(capture, "Take a foto!", (bar.shape[0] + 10,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0))

			capture[0:bar.shape[0], 0:bar.shape[1]] = bar
			cv2.imshow("Color analizator", capture)
			
		else:
			hist = centroid_histogram(model)
			bar = colors_img(hist, model.cluster_centers_)
			bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

			take_foto = get_percent(model)
			capture[0:bar.shape[0], 0:bar.shape[1]] = bar
			cv2.imshow("Color analizator", capture)
			thread_model = threading.Thread(target=train_model, name="train_model", args=[model, capture])
			thread_model.start()

		status, capture = cap.read()

		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break

	cv2.destroyAllWindows()
	cap.release()

