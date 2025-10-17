#!/usr/bin/env python3
import subprocess
import os
import cv2
import numpy as np

def run_command(command):
    """Выполняет команду и возвращает код возврата, stdout и stderr."""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def take_screenshot():
    """Снимает скриншот с подключенного устройства через ADB и сохраняет в текущей директории."""
    # Проверяем, подключено ли устройство
    code, out, err = run_command("adb devices")
    if code != 0 or "device" not in out:
        print("Ошибка: Устройство не подключено или ADB не работает.")
        print(f"ADB output: {out}")
        print(f"ADB error: {err}")
        return

    # Фиксированное имя файла
    filename = "screenshot.png"

    # Снимаем скриншот на устройстве
    print("Снимаем скриншот...")
    code, out, err = run_command(f"adb shell screencap -p /sdcard/{filename}")
    if code != 0:
        print(f"Ошибка при снятии скриншота: {err}")
        return

    # Скачиваем скриншот в текущую директорию
    print("Скачиваем скриншот...")
    code, out, err = run_command(f"adb pull /sdcard/{filename} .")
    if code != 0:
        print(f"Ошибка при скачивании: {err}")
        return

    # Удаляем скриншот с устройства
    run_command(f"adb shell rm /sdcard/{filename}")

    print(f"Скриншот сохранен как {filename} в директории {os.getcwd()}")

def analyze_screenshot():
    """Анализирует скриншот и определяет цвет первого человечка в очереди."""
    filename = "screenshot.png"
    if not os.path.exists(filename):
        print("Скриншот не найден.")
        return

    # Загружаем изображение
    image = cv2.imread(filename)
    if image is None:
        print("Не удалось загрузить изображение.")
        return

    height, width = image.shape[:2]

    # Определяем ROI для очереди (примерные проценты, нужно подстроить)
    roi_x1 = int(width * 0.3)  # 10% от ширины
    roi_y1 = int(height * 0.21)  # 20% от высоты
    roi_x2 = int(width * 0.8)  # 90% от ширины
    roi_y2 = int(height * 0.26)  # 80% от высоты
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Преобразуем в HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Сохраняем ROI для проверки
    cv2.imwrite("roi.png", roi)
    print("ROI сохранен как roi.png")

    # Сохраняем HSV изображение
    cv2.imwrite("hsv.png", hsv)
    print("HSV сохранен как hsv.png")

    S, V = 160, 200
    def m(h1, h2): return cv2.inRange(hsv, np.array([h1, S, V]), np.array([h2, 255, 255]))

    mask_y = m(20, 35)
    mask_b = m(90, 130)
    mask_r = cv2.bitwise_or(m(0, 10), m(170, 179))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_y, mask_b), mask_r)

    # Сохраняем объединенную маску
    cv2.imwrite("mask_combined.png", mask)
    print("Объединенная маска сохранена как mask_combined.png")

    # морфология
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)

    # Сохраняем маску после морфологии
    cv2.imwrite("mask_morph.png", mask)
    print("Маска после морфологии сохранена как mask_morph.png")

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Сохраняем изображение с контурами
    contour_img = roi.copy()
    cv2.drawContours(contour_img, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite("contours.png", contour_img)
    print("Контуры сохранены как contours.png")

    cands = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 80 or a > 5000:  # подстрой
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h <= w:  # человечки вертикальные
            continue

        # средний H внутри контура
        cmask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(cmask, [c], -1, 255, -1)

        # ядро контура
        core = cv2.erode(cmask, np.ones((5, 5), np.uint8), iterations=1)

        # подсчет пикселей каждого цвета в ядре
        ny = cv2.countNonZero(cv2.bitwise_and(mask_y, core))
        nb = cv2.countNonZero(cv2.bitwise_and(mask_b, core))
        nr = cv2.countNonZero(cv2.bitwise_and(mask_r, core))

        colors = ["yellow", "blue", "red"]
        color = colors[np.argmax([ny, nb, nr])]

        cands.append((x + roi_x1, y + roi_y1, w, h, color))

    if not cands:
        print("Человечки не найдены.")
        return

    # первый слева
    first = min(cands, key=lambda t: t[0])
    print("Цвет первого человечка:", first[4])

if __name__ == "__main__":
    take_screenshot()
    analyze_screenshot()