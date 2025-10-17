import cv2
img = cv2.imread("roi.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        h,s,v = hsv[y,x]
        print(f"HSV в точке ({x},{y}): {h},{s},{v}")

cv2.imshow("ROI", img)
cv2.setMouseCallback("ROI", show_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
