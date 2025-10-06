
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def _to_uint8(img):
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (np.clip(img,0,1)*255).astype(np.uint8)
        else:
            img = np.clip(img,0,255).astype(np.uint8)
    return img

def stack_h(imgL, imgR):
    h1,w1 = imgL.shape[:2]; h2,w2 = imgR.shape[:2]
    H = max(h1,h2); W = w1+w2
    out = np.zeros((H,W,3), dtype=np.uint8)
    out[:h1,:w1] = _to_uint8(imgL)
    out[:h2,w1:w1+w2] = _to_uint8(imgR)
    return out, w1

def draw_keypoints(image_path, pts, max_points=1200, r=2, color=(0,255,0)):
    im = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(im)
    P = pts[:max_points].astype(np.float32)
    for x,y in P:
        draw.ellipse((x-r, y-r, x+r, y+r), outline=color)
    return np.array(im)

def draw_matches(imgA_path, imgB_path, ptsA, ptsB, matches, inliers=None, max_draw=200):
    imgA = np.array(Image.open(imgA_path).convert('RGB'))
    imgB = np.array(Image.open(imgB_path).convert('RGB'))
    canvas, off = stack_h(imgA, imgB)
    im = Image.fromarray(canvas); draw = ImageDraw.Draw(im)
    use = matches[:max_draw]
    for k,(i,j) in enumerate(use):
        x1,y1 = ptsA[i]; x2,y2 = ptsB[j]
        x2 += off
        good = bool(inliers[k]) if (inliers is not None and k < len(inliers)) else None
        color = (0,255,0) if good else ((255,0,0) if good is not None else (255,255,0))
        draw.line([(float(x1),float(y1)),(float(x2),float(y2))], fill=color, width=1)
        draw.ellipse((x1-2,y1-2,x1+2,y1+2), outline=color)
        draw.ellipse((x2-2,y2-2,x2+2,y2+2), outline=color)
    return np.array(im)

def plot_hist(errors, title="Reprojection error (px)"):
    plt.figure(figsize=(6,3.5))
    plt.hist(errors, bins=50)
    plt.title(title); plt.xlabel("error (px)"); plt.ylabel("count")
    plt.show()

def _corners_polygon(w,h):
    return np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1],[0,0,1]], dtype=np.float64).T

def visualize_layout(files, H_to_ref):
    xs,ys=[],[]; polys=[]
    for k,p in enumerate(files):
        w,h = Image.open(p).size
        poly = H_to_ref[k] @ _corners_polygon(w,h)
        poly = poly[:2]/poly[2:]
        polys.append(poly)
        xs.extend(poly[0]); ys.extend(poly[1])
    xmin,xmax = min(xs), max(xs); ymin,ymax = min(ys), max(ys)
    plt.figure(figsize=(6,6))
    for poly in polys:
        plt.plot(poly[0], poly[1])
    plt.gca().invert_yaxis()
    plt.title("Projected image corners (layout)")
    plt.xlim([xmin, xmax]); plt.ylim([ymax, ymin])
    plt.show()

def compute_canvas_bounds(files, H_to_ref):
    xs,ys=[],[]
    for k,p in enumerate(files):
        w,h = Image.open(p).size
        poly = H_to_ref[k] @ _corners_polygon(w,h)
        poly = poly[:2]/poly[2:]
        xs.extend(poly[0]); ys.extend(poly[1])
    xmin,xmax = int(np.floor(min(xs))), int(np.ceil(max(xs)))
    ymin,ymax = int(np.floor(min(ys))), int(np.ceil(max(ys)))
    return xmin,ymin,xmax,ymax

def visualize_weight_maps(files, H_to_ref, sigma=15.0):
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from sift.gaussian import gaussian_blur
    from geom.warp import warp_inverse_map

    imgs = [np.array(Image.open(p).convert('RGB'), dtype=np.float64)/255.0 for p in files]

    xmin, ymin, xmax, ymax = compute_canvas_bounds(files, H_to_ref)
    Wc, Hc = xmax - xmin, ymax - ymin
    T_off = np.array([[1, 0, -xmin],
                      [0, 1, -ymin],
                      [0, 0,    1]], dtype=np.float64)

    weights = []
    for k, im in enumerate(imgs):
        Hc_i = T_off @ H_to_ref[k]
        warped = warp_inverse_map(im, Hc_i, (Hc, Wc), fill=0.0)

        mask = (warped.sum(axis=2) > 1e-6).astype(np.float64)
        w = gaussian_blur(mask, sigma=sigma)
        if w.max() > 0:
            w = w / w.max()
        weights.append(w)

        # show
        plt.figure(figsize=(4,3))
        plt.imshow(w, cmap='gray')
        plt.title(f'weight map #{k}')
        plt.axis('off')
        plt.show()

    return weights