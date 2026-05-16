import numpy as np

from predict.face_detector import detect_faces, draw_faces, extract_face


def test_detect_on_black_image_returns_no_face():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    faces = detect_faces(img)
    assert len(faces) == 0


def test_detect_accepts_grayscale_input():
    img = np.zeros((200, 200), dtype=np.uint8)
    faces = detect_faces(img)
    assert len(faces) == 0


def test_extract_face_shape_correct():
    img = np.arange(100 * 100 * 3).reshape(100, 100, 3).astype(np.uint8)
    bbox = (10, 20, 30, 40)
    face = extract_face(img, bbox)
    assert face.shape == (40, 30, 3)


def test_extract_face_content_correct():
    img = np.zeros((50, 50), dtype=np.uint8)
    img[10:20, 5:15] = 255
    bbox = (5, 10, 10, 10)
    face = extract_face(img, bbox)
    assert (face == 255).all()


def test_draw_faces_does_not_modify_input():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    original = img.copy()
    _ = draw_faces(img, [(10, 10, 30, 30)])
    assert np.array_equal(img, original)


def test_draw_faces_returns_modified_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    output = draw_faces(img, [(10, 10, 30, 30)])
    assert not np.array_equal(img, output)