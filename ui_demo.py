from PySide6.QtWidgets import (
	QApplication, QPushButton,
	QLabel, QWidget, QVBoxLayout,
	QLineEdit,QGraphicsScene,QFileDialog,QMessageBox,QGraphicsView,QGraphicsLineItem,
)

from PySide6.QtCore import QDir, QFile, QIODevice, QTextStream, Qt, QLineF, QRectF
from PySide6.QtGui import QPainter, QPixmap, QPen, QColor, QBrush, QImage

from demo import Ui_Form


import numpy as np
from PIL import Image, ImageQt
import torch
import cv2
from src.funcs import ldm_sample_folder
from src.funcs.ldm_sample_folder import run
from src.postprocess import seamless_clone_face
from src.postprocess.seamless_clone_face import run_seamless_clone_face
from src.postprocess import seamless_clone_hand
from src.postprocess.seamless_clone_hand import run_seamless_clone_hand

palette = [
    (0, 0, 0),
    (255, 250, 250),
    (220, 220, 220),
    (250, 235, 215),
    (255, 250, 205),
    (211, 211, 211),
    (70, 130, 180),
    (127, 255, 212),
    (0, 100, 0),
    (50, 205, 50),
    (255, 255, 0),
    (245, 222, 179),
    (255, 140, 0),
    (255, 0, 0),
    (16, 78, 139),
    (144, 238, 144),
    (50, 205, 174),
    (50, 155, 250),
    (160, 140, 88),
    (213, 140, 88),
    (90, 140, 90),
    (185, 210, 205),
    (130, 165, 180),
    (225, 141, 151)
]

color_list = [
    QColor(0, 0, 0),
    QColor(255, 250, 250),
    QColor(220, 220, 220),
    QColor(250, 235, 215),
    QColor(255, 250, 205),
    QColor(211, 211, 211),
    QColor(70, 130, 180),
    QColor(127, 255, 212),
    QColor(0, 100, 0),
    QColor(50, 205, 50),
    QColor(255, 255, 0),
    QColor(245, 222, 179),
    QColor(255, 140, 0),
    QColor(255, 0, 0),
    QColor(16, 78, 139),
    QColor(144, 238, 144),
    QColor(50, 205, 174),
    QColor(50, 155, 250),
    QColor(160, 140, 88),
    QColor(213, 140, 88),
    QColor(90, 140, 90),
    QColor(185, 210, 205),
    QColor(130, 165, 180),
    QColor(225, 141, 151)
]


class GraphicsScene(QGraphicsScene):
    def __init__(self, mode, size, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.mode = mode
        self.size = size
        self.mouse_clicked = False
        self.prev_pt = None

        # self.masked_image = None

        # save the points
        self.mask_points = []
        for i in range(len(color_list)):
            self.mask_points.append([])

        # save the size of points
        self.size_points = []
        for i in range(len(color_list)):
            self.size_points.append([])

        # save the history of edit
        self.history = []

    def mousePressEvent(self, event):
        self.mouse_clicked = True
        print("mouse clicked")

    def mouseReleaseEvent(self, event):
        self.prev_pt = None
        self.mouse_clicked = False
        print("mouse released")

    def mouseMoveEvent(self, event):  # drawing
        if self.mouse_clicked:
            if self.prev_pt:
                self.drawMask(self.prev_pt, event.scenePos(),
                              color_list[self.mode], self.size)
                print(self.mode)
                pts = {}
                pts['prev'] = (int(self.prev_pt.x()), int(self.prev_pt.y()))
                pts['curr'] = (int(event.scenePos().x()),
                               int(event.scenePos().y()))

                self.size_points[self.mode].append(self.size)
                self.mask_points[self.mode].append(pts)
                self.history.append(self.mode)
                self.prev_pt = event.scenePos()
            else:
                self.prev_pt = event.scenePos()

    def drawMask(self, prev_pt, curr_pt, color, size):
        lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        lineItem.setPen(QPen(color, size, Qt.SolidLine))  # rect
        self.addItem(lineItem)


class MainWindow(QWidget, Ui_Form):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setStyleSheet(f"QWidget {{background-color: #FFFFFF; color: #333333; }} QPushButton:hover {{ background-color: #333333; color: #FFFFFF; }}")
		self.pushButton_load_parsing.clicked.connect(self.load_parsing)
		self.pushButton_load_pose.clicked.connect(self.load_pose)
		self.pushButton_generate.clicked.connect(self.generate)
		self.pushButton_clear.clicked.connect(self.clear)
		self.pushButton_save_output.clicked.connect(self.save_output)
		self.pushButton_save_parsing.clicked.connect(self.save_parsing)

		self.pushButton_background.clicked.connect(self.set_mode_background)
		self.pushButton_top.clicked.connect(self.set_mode_top)
		self.pushButton_outer.clicked.connect(self.set_mode_outer)
		self.pushButton_skirt.clicked.connect(self.set_mode_skirt)
		self.pushButton_dress.clicked.connect(self.set_mode_dress)
		self.pushButton_pants.clicked.connect(self.set_mode_pants)
		self.pushButton_leggings.clicked.connect(self.set_mode_leggings)
		self.pushButton_headwear.clicked.connect(self.set_mode_headwear)
		self.pushButton_eyeglass.clicked.connect(self.set_mode_eyeglass)
		self.pushButton_neckwear.clicked.connect(self.set_mode_neckwear)
		self.pushButton_belt.clicked.connect(self.set_mode_belt)
		self.pushButton_footwear.clicked.connect(self.set_mode_footwear)
		self.pushButton_bag.clicked.connect(self.set_mode_bag)
		self.pushButton_hair.clicked.connect(self.set_mode_hair)
		self.pushButton_face.clicked.connect(self.set_mode_face)
		self.pushButton_skin.clicked.connect(self.set_mode_skin)
		self.pushButton_ring.clicked.connect(self.set_mode_ring)
		self.pushButton_wristwear.clicked.connect(self.set_mode_wrist)
		self.pushButton_socks.clicked.connect(self.set_mode_socks)
		self.pushButton_gloves.clicked.connect(self.set_mode_gloves)
		self.pushButton_necklace.clicked.connect(self.set_mode_necklace)
		self.pushButton_rompers.clicked.connect(self.set_mode_rompers)
		self.pushButton_earrings.clicked.connect(self.set_mode_earrings)
		self.pushButton_tie.clicked.connect(self.set_mode_tie)

		self.state = 0
		self.parsing = None
		self.pose = None
		self.text = None
		self.output = None

		# setting brush state and size
		self.mode = 0
		self.size = 6
		self.mouse_clicked = False
		self.prev_pt = None

		scene_parsing = QGraphicsScene()
		self.graphicsView_parsing.setScene(scene_parsing)
		self.graphicsView_parsing.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		self.graphicsView_parsing.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.graphicsView_parsing.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		scene_pose = QGraphicsScene()
		self.graphicsView_pose.setScene(scene_pose)
		self.graphicsView_pose.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		self.graphicsView_pose.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.graphicsView_pose.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		scene_output = QGraphicsScene()
		self.graphicsView_output.setScene(scene_output)
		self.graphicsView_output.setAlignment(Qt.AlignTop | Qt.AlignLeft)
		self.graphicsView_output.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
		self.graphicsView_output.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self.model = None
		self.vae = None

	def load_parsing(self):
		fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
		if fileName:
			scene_parsing = GraphicsScene(self.mode, self.size)
			image = Image.open(fileName).resize((256, 512))
			qimage = ImageQt.ImageQt(image)
			image = QPixmap.fromImage(qimage)
			image = image.scaled(256, 512)
			scene_parsing.setSceneRect(0, 0, 256, 512)
			scene_parsing.addPixmap(image)

			self.graphicsView_parsing.setScene(scene_parsing)
			self.parsing = ImageQt.fromqimage(qimage)

	def load_pose(self):
		fileName, _ = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
		if fileName:
			scene = QGraphicsScene()
			image = QPixmap(fileName)
			image = image.scaled(256, 512)
			scene.setSceneRect(0, 0, 256, 512)
			scene.addPixmap(image)
			self.graphicsView_pose.setScene(scene)
			self.pose = Image.open(fileName)

	def generate(self):
		self.auto_save_parsing()
		self.auto_save_pose()
		self.auto_save_text()
		self.model, self.vae = run(self.model, self.vae)
		scene = QGraphicsScene()
		image = QPixmap(f"./src/deepfashion/cond_text_image_samples/result.png")
		image = image.scaled(256, 512)
		scene.setSceneRect(0, 0, 256, 512)
		scene.addPixmap(image)
		self.graphicsView_output.setScene(scene)

	def save_output(self):
		fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
		if fileName:
			rect = self.graphicsView_output.scene().sceneRect()
			image = QImage(rect.width(), rect.height(), QImage.Format_RGB888)
			painter = QPainter(image)
			self.graphicsView_output.render(painter)
			painter.end()
			image.save(fileName)
			QMessageBox.information(self, "Save Image", "Image saved successfully.")

	def make_mask(self, mask, pts, sizes, color):
		if len(pts) > 0:
			for idx, pt in enumerate(pts):
				cv2.line(mask, pt['prev'], pt['curr'], color, sizes[idx])
		return mask

	def find_closest_color_index(self, pixel, palette):
		distances = [np.sqrt(np.sum((np.array(pixel) - np.array(color)) ** 2)) for color in palette]
		return np.argmin(distances)
	def find_color_index(self, pixel, palette):
		for index, color in enumerate(palette):
			if np.all(color == pixel):
				return index
		return -1
	def save_parsing(self):
		fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
		if fileName:
			self.parsing = np.array(self.parsing)
			for i in range(24):
				self.parsing = self.make_mask(self.parsing, self.graphicsView_parsing.scene().mask_points[i], self.graphicsView_parsing.scene().size_points[i], palette[i])
			index_image = np.zeros((self.parsing.shape[0], self.parsing.shape[1]), dtype=np.uint8)
			for i in range(self.parsing.shape[0]):
				for j in range(self.parsing.shape[1]):
					index_image[i, j] = self.find_color_index(self.parsing[i, j], palette)
					if index_image[i, j] == -1:
						index_image[i, j] = self.find_closest_color_index(self.parsing[i, j], palette)

			index_image = Image.fromarray(index_image, 'P')
			flattened_palette = sum(palette, ())
			index_image.putpalette(flattened_palette)
			index_image.save(fileName)
			QMessageBox.information(self, "Save Image", "Image saved successfully.")

	def auto_save_parsing(self):
		self.parsing = np.array(self.parsing)
		for i in range(24):
			self.parsing = self.make_mask(self.parsing, self.graphicsView_parsing.scene().mask_points[i], self.graphicsView_parsing.scene().size_points[i], palette[i])
		index_image = np.zeros((self.parsing.shape[0], self.parsing.shape[1]), dtype=np.uint8)
		for i in range(self.parsing.shape[0]):
			for j in range(self.parsing.shape[1]):
				index_image[i, j] = self.find_color_index(self.parsing[i, j], palette)
				if index_image[i, j] == -1:
					index_image[i, j] = self.find_closest_color_index(self.parsing[i, j], palette)

		index_image = Image.fromarray(index_image, 'P')
		flattened_palette = sum(palette, ())
		index_image.putpalette(flattened_palette)
		index_image.resize((512, 1024)).save(f"./src/deepfashion/cond_text_image_samples/parsing.png")

	def auto_save_pose(self):
		self.pose.save(f"./src/deepfashion/cond_text_image_samples/pose.png")

	def auto_save_text(self):
		self.text = self.lineEdit.text()
		with open(f"./src/deepfashion/cond_text_image_samples/text.txt", "w") as file:
			file.write(self.text)

	def clear(self):
		scene_parsing = QGraphicsScene()
		scene_pose = QGraphicsScene()
		scene_output = QGraphicsScene()
		self.graphicsView_parsing.setScene(scene_parsing)
		self.graphicsView_pose.setScene(scene_pose)
		self.graphicsView_output.setScene(scene_output)
		self.state = 0

	def set_mode_background(self):
		self.graphicsView_parsing.scene().mode = 0
	def set_mode_top(self):
		self.graphicsView_parsing.scene().mode = 1

	def set_mode_outer(self):
		self.graphicsView_parsing.scene().mode = 2

	def set_mode_skirt(self):
		self.graphicsView_parsing.scene().mode = 3

	def set_mode_dress(self):
		self.graphicsView_parsing.scene().mode = 4

	def set_mode_pants(self):
		self.graphicsView_parsing.scene().mode = 5

	def set_mode_leggings(self):
		self.graphicsView_parsing.scene().mode = 6

	def set_mode_headwear(self):
		self.graphicsView_parsing.scene().mode = 7

	def set_mode_eyeglass(self):
		self.graphicsView_parsing.scene().mode = 8

	def set_mode_neckwear(self):
		self.graphicsView_parsing.scene().mode = 9

	def set_mode_belt(self):
		self.graphicsView_parsing.scene().mode = 10

	def set_mode_footwear(self):
		self.graphicsView_parsing.scene().mode = 11

	def set_mode_bag(self):
		self.graphicsView_parsing.scene().mode = 12

	def set_mode_hair(self):
		self.graphicsView_parsing.scene().mode = 13

	def set_mode_face(self):
		self.graphicsView_parsing.scene().mode = 14

	def set_mode_skin(self):
		self.graphicsView_parsing.scene().mode = 15

	def set_mode_ring(self):
		self.graphicsView_parsing.scene().mode = 16

	def set_mode_wrist(self):
		self.graphicsView_parsing.scene().mode = 17

	def set_mode_socks(self):
		self.graphicsView_parsing.scene().mode = 18

	def set_mode_gloves(self):
		self.graphicsView_parsing.scene().mode = 19

	def set_mode_necklace(self):
		self.graphicsView_parsing.scene().mode = 20

	def set_mode_rompers(self):
		self.graphicsView_parsing.scene().mode = 21

	def set_mode_earrings(self):
		self.graphicsView_parsing.scene().mode = 22

	def set_mode_tie(self):
		self.graphicsView_parsing.scene().mode = 23


	def mousePressEvent(self, event):
		self.mouse_clicked = True
		pos = event.pos()
		x, y = pos.x(), pos.y()
		self.prev_pt = event.pos()
		print(x, y)

	def mouseReleaseEvent(self, event):
		self.mouse_clicked = False

if __name__ == '__main__':
	app = QApplication([])
	window = MainWindow()
	window.show()
	app.exec()

