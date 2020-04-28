import cv2
import numpy as np

def preprocessing(img):
	resized = cv2.resize(img,(600,800))
	gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
	blurr = cv2.GaussianBlur(gray,(5,5),0)
	edge = cv2.Canny(blurr,0,50)

	contours,_ = cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours,key = cv2.contourArea,reverse = True)

	for i in contours:
		elip = cv2.arcLength(i,True)
		approx = cv2.approxPolyDP(i,0.08 * elip, True)

		if len(approx) == 4:
			doc = approx 
			break

	cv2.drawContours(img,[doc], -1, (0,255,0), 2)
	doc = doc.reshape((4,2))
	new_doc = np.zeros((4,2),dtype = "float32")
	Sum = doc.sum(axis = 1)
	new_doc[0] = doc[np.argmin(Sum)]
	new_doc[2] = doc[np.argmax(Sum)]
	Diff = np.diff(doc,axis = 1)
	new_doc[1] = doc[np.argmin(Diff)]
	new_doc[3] = doc[np.argmax(Diff)]
	(tl,tr,br,bl) = new_doc
	dist1 = np.linalg.norm(br-bl)
	dist2 = np.linalg.norm(tr-tl)
	maxLen = max(int(dist1),int(dist2))
	dist3 = np.linalg.norm(tr-br)
	dist4 = np.linalg.norm(tl-bl)
	maxHeight = max(int(dist3), int(dist4))
	dst = np.array([[0,0],[maxLen-1, 0],[maxLen-1, maxHeight-1], [0, maxHeight-1]], dtype="float32")
	N = cv2.getPerspectiveTransform(new_doc, dst)
	warp = cv2.warpPerspective(img, N, (maxLen, maxHeight))
	img2 = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
	img2 = cv2.resize(img2,(600,800))
	return img2


if __name__ == "__main__":
	image = cv2.imread("test-2.jpg",1)
	resized = cv2.resize(image,(image.shape[1],image.shape[0]))
	cv2.imshow("image",preprocessing(resized))
	cv2.waitKey(0)
	cv2.destroyAllWindows()