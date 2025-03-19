import os
import cv2

classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
			'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'people', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

dir = "ExDark/"
dir2 = "ExDark_Annno/"

testDict = []
annoOpen = open(dir2+"/imageinfolist.txt", "r")
lines = annoOpen.readlines()
for line in lines:
	data = line.split(" ")
	if data[4].replace("\n", "") == "3":
		testDict.append(data[0])

print(testDict)

for root, dirs, files in os.walk(dir):
	for name in files:
		print(name)
		if not ("jpg" in name or "png" in name or "JPEG" in name 
			or "JPG" in name or "jpeg" in name):
			continue

		if ("images" in root or "labels" in root):
			continue

		if not (name in testDict):
			continue

		print(os.path.join(root, name))

		filename = os.path.join(root, name)
		img = cv2.imread(filename)
		h, w, c = img.shape

		label = filename.replace("ExDark_all", "ExDark_Annno")+".txt"
		annoOpen = open(label, "r")

		outImg = dir+"/images/test/"+name.split(".")[0] + ".png"
		annoOut = open(outImg.replace("images","labels").replace(".png",".txt"), "w")

		lines = annoOpen.readlines()
		for i in range(1, len(lines)):
			data = lines[i].split(" ")
			print(data[0])

			try:
				class_ind = classes.index(data[0].lower().strip())
			except:
				continue

			x = (float(data[1])+float(data[3])/2.0)/float(w)
			y = (float(data[2])+float(data[4])/2.0)/float(h)
			width = float(data[3])/float(w)
			height = float(data[4])/float(h)
			annoOut.write(str(class_ind) + " " + str(x) + " " + str(y) + " " + str(width) + " " + str(height))
			annoOut.write("\n")

		annoOut.close()
		cv2.imwrite(outImg, img)

		print(lines)