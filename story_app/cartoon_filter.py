import cv2
import numpy as np

def cartoonize(input_path, output_path, variation=0):
    """
    Disney-Pixar 3D Animator Synth 3.0: High Clarity & Cinematic Depth.
    """
    img = cv2.imread(input_path)
    if img is None: return False

    h, w = img.shape[:2]
    
    # 1. Background Variety Logic (Artificial Panning/Zooming)
    # This makes every page look like a different "shot" of the movie
    zoom = 1.0 - (variation * 0.04) % 0.15 # subtle zoom up to 15%
    nh, nw = int(h * zoom), int(w * zoom)
    
    # Cycle through different framing: Center, Top-Left, Bottom-Right, etc.
    framing = [
        ( (h-nh)//2, (w-nw)//2 ), # Center
        ( 0, 0 ),                 # Top-Left
        ( h-nh, w-nw ),           # Bottom-Right
        ( 0, w-nw ),              # Top-Right
        ( h-nh, 0 ),              # Bottom-Left
    ]
    y1, x1 = framing[variation % len(framing)]
    
    img = img[y1:y1+nh, x1:x1+nw]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Pixar-Style Sculpting (Edge Preserving Filter)
    # This is much sharper and cleaner than Bilateral filtering
    # sigma_s=10 is spatial smoothness, sigma_r=0.4 is range (color) smoothness
    smooth = cv2.edgePreservingFilter(img, flags=1, sigma_s=15, sigma_r=0.3)
    
    # 3. Cinematic Detail Boost (Makes eyes/hair pop)
    smooth = cv2.detailEnhance(smooth, sigma_s=10, sigma_r=0.15)

    # 4. Pixar Color Grading (Dramatic Moods)
    moods = [
        (1.25, 1.05, 0.8, 15),   # 0: Golden Kingdom (Morning)
        (0.8, 1.35, 0.9, -10),   # 1: Enchanted Forest (Emerald)
        (0.7, 0.8, 1.45, -35),   # 2: Crystal Caverns (Deep Blue)
        (1.1, 0.9, 1.4, 10),     # 3: Magic Twilight (Purple)
        (1.3, 1.25, 1.1, 30),    # 4: Royal Celebration (Vibrant)
    ]
    rm, gm, bm, br = moods[variation % len(moods)]
    
    b, g, r = cv2.split(smooth)
    b = cv2.convertScaleAbs(b, alpha=bm)
    g = cv2.convertScaleAbs(g, alpha=gm)
    r = cv2.convertScaleAbs(r, alpha=rm)
    final = cv2.merge([b, g, r])
    
    # Boost saturation and brightness
    final = cv2.convertScaleAbs(final, alpha=1.2, beta=br)

    # 5. HIGH-DYNAMICS CONTRAST (CLAHE) - Essential for clarity
    lab = cv2.cvtColor(final, cv2.COLOR_BGR2LAB)
    l, a, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    final = cv2.merge((l, a, b_chan))
    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    # 6. Ultra-Sharp Finish (Unsharp Mask)
    # Removes the "blurry" look the user complained about
    blurred = cv2.GaussianBlur(final, (0, 0), 3)
    final = cv2.addWeighted(final, 1.7, blurred, -0.7, 0)

    # Save with maximum quality
    cv2.imwrite(output_path, final, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"DEBUG: Saved Pixar Synth 3.0 (Variation {variation}) to {output_path}")
    return True
