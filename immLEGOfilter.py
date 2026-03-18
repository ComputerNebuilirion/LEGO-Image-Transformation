import cv2
import numpy as np
import time
import collections
class Brick:
    def __init__(self, x, y, width, height, color, color_name):
        self.x = x 
        self.y = y
        self.width = width
        self.height = height
        self.color = color  # (B, G, R)
        if isinstance(color, np.ndarray):
            self.color = tuple(int(c) for c in color)
        else:
            self.color = tuple(color)
        self.color_name = color_name
    
    def size_str(self):
        return f"{self.width}x{self.height}"
    
    def area(self):
        return self.width * self.height
    

class LegoColorPalette:
    
    # In BGR format
    COLORS = {
        'white': (255, 255, 255),
        'light_gray': (192, 192, 192),
        'gray': (128, 128, 128),
        'dark_gray': (64, 64, 64),
        'black': (0, 0, 0),
        'red': (205, 0, 0),
        'light_nougat': (230, 190, 140),
        'nougat': (200, 160, 100),
        'dark_red': (139, 0, 0), 
        'orange': (255, 140, 0), 
        'yellow': (255, 255, 0),
        'warm_yellow': (255, 230, 20),
        'olive': (140, 180, 0),
        'lime': (0, 255, 0),
        'green': (0, 128, 0),
        'dark_green': (0, 100, 0),
        'cyan': (0, 255, 255), 
        'blue': (0, 0, 255), 
        'dark_blue': (0, 0, 139),
        'purple': (255, 0, 255),
        'pink': (255, 192, 203),
        'brown': (139, 69, 19),
        'tan': (224, 176, 128),
    }
    
    # Available brick sizes (width, height) in studs
    BRICK_SIZES = [(4, 2), (2, 4), (2, 2), (4, 1), (1, 4), (2, 1), (1, 2), (1, 1)]
    
    @classmethod
    def find_closest(cls, bgr_color):
        min_dist = float('inf')
        closest_name = 'black'
        closest_color = (0, 0, 0)
        
        for name, color in cls.COLORS.items():
            dist = (bgr_color[0] - color[0]) ** 2 + \
                   (bgr_color[1] - color[1]) ** 2 + \
                   (bgr_color[2] - color[2]) ** 2
            if dist < min_dist:
                min_dist = dist
                closest_name = name
                closest_color = color
        
        return closest_name, closest_color
    
def white_balance_out(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        balanced = cv2.merge([l, a, b])
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

class GridQuantizer:
    def __init__(self, grid_size=4):
        self.grid_size = grid_size
    
    def white_balance(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        balanced = cv2.merge([l, a, b])
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

    def quantize(self, img):
        # Preprocessing
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = self.white_balance(img)
        img = cv2.medianBlur(img, 3)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)

        h, w = img.shape[:2]
        
        new_h = (h // self.grid_size) * self.grid_size
        new_w = (w // self.grid_size) * self.grid_size
        img = cv2.resize(img, (new_w, new_h))
        
        small_h = new_h // self.grid_size
        small_w = new_w // self.grid_size
        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        pixels = small.reshape(-1, 3).astype(np.float32)
        
        color_names_list = list(LegoColorPalette.COLORS.keys())
        color_values = np.array(list(LegoColorPalette.COLORS.values()), dtype=np.float32)
        
        dists = np.linalg.norm(pixels[:, None, :] - color_values[None, :, :], axis=2)
        
        closest_idx = dists.argmin(axis=1)
        
        grid = color_values[closest_idx].reshape(small_h, small_w, 3).astype(np.uint8)
        
        color_names = [[color_names_list[closest_idx[i * small_w + j]] 
                    for j in range(small_w)] for i in range(small_h)]
        
        return grid, color_names, small
    

class BrickMerger:
    def __init__(self):
        self.visited = None
        self.grid = None
        self.color_names = None
        self.bricks = []
    
    def merge(self, grid, color_names):
        self.grid = grid
        self.color_names = color_names
        h, w = grid.shape[:2]
        self.visited = [[False for _ in range(w)] for _ in range(h)]
        self.bricks = []
        
        for i in range(h):
            for j in range(w):
                if self.visited[i][j]:
                    continue
                
                color = tuple(grid[i, j])
                color_name = color_names[i][j]
                
                placed = False
                for bw, bh in LegoColorPalette.BRICK_SIZES:
                    if self._can_place(i, j, bw, bh, color):
                        self._place_brick(i, j, bw, bh, color, color_name)
                        placed = True
                        break
                
                if not placed:
                    self._place_brick(i, j, 1, 1, color, color_name)
        
        return self.bricks
    
    def _can_place(self, y, x, w, h, color):
        gh, gw = self.grid.shape[:2]
        
        if x + w > gw or y + h > gh:
            return False
        
        for i in range(y, y + h):
            for j in range(x, x + w):
                if self.visited[i][j]:
                    return False
                if tuple(self.grid[i, j]) != color:
                    return False
        return True
    
    def _place_brick(self, y, x, w, h, color, color_name):
        self.bricks.append(Brick(x, y, w, h, color, color_name))
        
        for i in range(y, y + h):
            for j in range(x, x + w):
                self.visited[i][j] = True


class LegoRenderer:
    def __init__(self, brick_pixel_size=20):
        self.brick_size = brick_pixel_size
    
    def render(self, bricks, grid_h, grid_w):
        out_h = grid_h * self.brick_size
        out_w = grid_w * self.brick_size
        output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        for brick in bricks:
            self._render_brick(output, brick)
        
        return output
    
    def _render_brick(self, canvas, brick):
        x = brick.x * self.brick_size
        y = brick.y * self.brick_size
        w = brick.width * self.brick_size
        h = brick.height * self.brick_size

        color = tuple(int(c) for c in brick.color)
        
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        
        border_color = (50, 50, 50)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_color, 2)

        
        self._render_studs(canvas, x, y, w, h, brick.color)
    
    def _render_studs(self, canvas, x, y, w, h, color):
        radius = self.brick_size // 3
        
        if not isinstance(color, tuple):
            color = tuple(int(c) for c in color)
        
        cols = max(1, w // self.brick_size)
        rows = max(1, h // self.brick_size)
        
        for i in range(rows):
            for j in range(cols):
                cx = x + (j * self.brick_size) + self.brick_size // 2
                cy = y + (i * self.brick_size) + self.brick_size // 2
                
                stud_color = (
                    min(255, int(color[0] * 1.1)),
                    min(255, int(color[1] * 1.1)),
                    min(255, int(color[2] * 1.1))
                )
                cv2.circle(canvas, (cx, cy), radius, stud_color, -1)
                cv2.circle(canvas, (cx, cy), radius, (30, 30, 30), 1)
                cv2.ellipse(canvas, (cx, cy), (radius, radius), 0, 45, 135, (30, 30, 30), 2)


class LegoConverter:
    def __init__(self, grid_size=4, brick_pixel_size=20):
        self.quantizer = GridQuantizer(grid_size)
        self.merger = BrickMerger()
        self.renderer = LegoRenderer(brick_pixel_size)
        self.grid_size = grid_size
    
    def convert(self, img):
        grid, color_names, small = self.quantizer.quantize(img)
        grid_h, grid_w = grid.shape[:2]
        
        #print(f"Grid size: {grid_w}x{grid_h} ({grid_w * grid_h} cells)")
        
        bricks = self.merger.merge(grid, color_names)
        
        rendered = self.renderer.render(bricks, grid_h, grid_w)
        
        stats = self._calculate_stats(bricks)
        
        return rendered, bricks, stats
    
    def _calculate_stats(self, bricks):
        stats = {
            'total_bricks': len(bricks),
            'total_stud_count': sum(b.area() for b in bricks),
            'by_size': {},
            'by_color': {},
        }
        
        for brick in bricks:
            size_str = brick.size_str()
            if size_str not in stats['by_size']:
                stats['by_size'][size_str] = 0
            stats['by_size'][size_str] += 1
            
            if brick.color_name not in stats['by_color']:
                stats['by_color'][brick.color_name] = 0
            stats['by_color'][brick.color_name] += 1
        
        return stats
    
    def print_summary(self, stats):
        print("\n" + "="*50)
        print("LEGO Brick Summary")
        print("="*50)
        print(f"Total bricks: {stats['total_bricks']}")
        print(f"Total studs (1x1 equivalent): {stats['total_stud_count']}")
        
        print("\nBy Size:")
        size_items = []
        for size, count in stats['by_size'].items():
            w, h = map(int, size.split('x'))
            size_items.append((w*h, size, count))
        size_items.sort(reverse=True)
        
        for _, size, count in size_items:
            print(f"  {size}: {count}")
        
        print("\nBy Color:")
        color_items = list(stats['by_color'].items())
        color_items.sort(key=lambda x: -x[1])
        
        for color, count in color_items:
            print(f"  {color}: {count}")

import threading
import queue

class LegoCamera:
    def __init__(self, grid_size=8, brick_pixel_size=20, threshold=0.5):
        self.converter = LegoConverter(grid_size, brick_pixel_size)
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.running = False
        self.process_thread = None
        self.threshold = threshold
    
    def start(self):
        self.running = True
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.start()
    
    def stop(self):
        self.running = False
        if self.process_thread:
            self.process_thread.join()
    
    def _process_loop(self):
        while self.running:
            try:
                frame = self.input_queue.get(timeout=0.1)
                lego, _, _ = self.converter.convert(frame)
                if self.output_queue.full():
                    self.output_queue.get()
                self.output_queue.put(lego)
            except queue.Empty:
                continue
    
    def process_frame(self, frame):
        if self.input_queue.full():
            self.input_queue.get()
        self.input_queue.put(frame)
        
        if not self.output_queue.empty():
            return self.output_queue.get()
        return None


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

camera = LegoCamera(grid_size=8, brick_pixel_size=20)
camera.start()

fps_history = collections.deque(maxlen=30)
prev_time = time.time()
font = cv2.FONT_HERSHEY_SIMPLEX

LEGO_ON = False
WHITE_ON = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.flip(frame, 1)
    small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    if LEGO_ON:
    
        rendered = camera.process_frame(small)

        if rendered is not None:
            rendered = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
            
            display = cv2.resize(rendered, (960, 720))

            current_time = time.time()
            delta = current_time - prev_time
            prev_time = current_time
            
            if delta > 0:
                instant_fps = 1.0 / delta
                fps_history.append(instant_fps)
                avg_fps = sum(fps_history) / len(fps_history)
                
                cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 30), 
                            font, 1, (0, 255, 0), 2)
            
            cv2.imshow('LEGO Filter', display)
    
    else:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        small = cv2.resize(small, (960, 720))
        if WHITE_ON:
            small = white_balance_out(small)
        cv2.imshow("LEGO Filter", small)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        LEGO_ON = not LEGO_ON
    elif key == ord('w'):
        WHITE_ON = not WHITE_ON
    

camera.stop()
cap.release()
cv2.destroyAllWindows()
