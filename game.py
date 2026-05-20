#!/usr/bin/env python3
"""
High Fantasy Word Explorer

Explore AI-generated scenes and guess the word each one depicts.
Press Tab to open the input field, type your answer, press Enter.
Find all 10 words to win!

Usage:
    python game.py
    OVIE_PATH=/path/to/ovie python game.py

Controls:
    Z / S        – move forward / backward
    Left / Right – yaw (turn left / right)
    Up / Down    – pitch (look up / down)
    Tab          – open / close word-input field
    Enter        – submit guess
    Backspace    – delete last character
    R            – reset camera to origin
    ESC / Q      – close input field / quit
"""

import argparse
import os
import random
import sys
import threading

import numpy as np
import pygame
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# ── Locate the OVIE repository ────────────────────────────────────────────────
_ovie_path = os.environ.get(
    "OVIE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ovie"),
)
if not os.path.isdir(_ovie_path):
    sys.exit(
        f"OVIE repo not found at '{_ovie_path}'.\n"
        "Clone https://github.com/kyutai-labs/ovie and set OVIE_PATH=<path>."
    )
sys.path.insert(0, _ovie_path)

from models.models import OVIEModel  # noqa: E402
from utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402
from miro import MiroPipeline  # noqa: E402

# ── Word list (30 entries, 10 chosen per game) ───────────────────────────────
WORDS_AND_PROMPTS = [
    # scenes ──────────────────────────────────────────────────────────────────
    ("CASTLE",
     "A grand medieval stone castle rising from a misty sea cliff at golden hour, tall towers "
     "with glowing windows, a drawbridge over a dark moat, ravens circling, high fantasy, "
     "photorealistic"),
    ("FOREST",
     "An ancient enchanted forest with enormous silver-barked trees, bioluminescent mushrooms "
     "and glowing flowers on the mossy floor, magical golden light rays filtering through the "
     "dense canopy, high fantasy, photorealistic"),
    ("DUNGEON",
     "A dark underground dungeon with rough stone walls and iron-barred cells, flickering "
     "torchlight casting long shadows on bones and rusted weapons on the damp floor, heavy "
     "chains on the walls, gothic high fantasy, photorealistic"),
    ("TAVERN",
     "Interior of a warm medieval fantasy tavern, low wooden beams, a large stone fireplace "
     "roaring with fire, cloaked adventurers at oak tables with tankards of ale, warm "
     "candlelight, high fantasy, photorealistic"),
    ("PORTAL",
     "A swirling circular arcane portal of violet and gold energy suspended between ancient "
     "moss-covered stone pillars in a ruin, glimpses of another realm through the gateway, "
     "glowing runes carved into the stone, high fantasy, photorealistic"),
    ("THRONE",
     "An imposing dark throne room with a massive obsidian throne on a raised dais, towering "
     "pillars lined with burning braziers, tattered battle banners hanging from vaulted "
     "ceilings, dramatic candlelight, high fantasy, photorealistic"),
    ("RUINS",
     "Ancient stone ruins of a fallen elven city overgrown with vines and glowing moss, "
     "crumbling archways and broken statues, mist drifting through, high fantasy, "
     "photorealistic"),
    ("CRYPT",
     "A vast underground crypt with rows of stone sarcophagi carved with warrior reliefs, "
     "flickering torch sconces on damp walls, cobwebs and scattered bones, high fantasy, "
     "photorealistic"),
    ("SHRINE",
     "A moss-covered outdoor forest shrine with a stone idol surrounded by offerings of "
     "candles and flowers, shafts of magical dappled light, ancient mystical atmosphere, "
     "high fantasy, photorealistic"),
    ("ALTAR",
     "A dark stone altar in an underground ritual chamber, carved with arcane symbols, "
     "surrounded by burning black candles, ominous light from above, high fantasy, "
     "photorealistic"),
    ("FORGE",
     "Interior of a dwarven forge, a massive furnace blazing with orange fire, sparks flying, "
     "glowing hot iron on a huge anvil, weapons and tools hung on stone walls, high fantasy, "
     "photorealistic"),
    # objects — close up ───────────────────────────────────────────────────────
    ("POTION",
     "Close up of a cluttered alchemist workshop with glowing coloured potions in glass vials "
     "and flasks, a bubbling cauldron emitting rainbow smoke, dried herbs, an open ancient "
     "grimoire, candlelight, high fantasy, photorealistic"),
    ("CROWN",
     "Close up of an ancient royal crown wrought from dark twisted gold, set with glowing "
     "rubies and sapphires, resting on red velvet, dramatic side lighting, high fantasy, "
     "photorealistic"),
    ("GRIMOIRE",
     "Close up of an open ancient spellbook with yellowed pages covered in glowing arcane "
     "symbols and diagrams, a quill pen resting on the page, flickering candlelight, high "
     "fantasy, photorealistic"),
    ("SWORD",
     "Close up of a legendary enchanted sword with a jewelled crossguard, glowing runes etched "
     "along the gleaming blade, embedded in a mossy stone, dramatic lighting, high fantasy, "
     "photorealistic"),
    # persons and creatures — portrait ────────────────────────────────────────
    ("DRAGON",
     "Portrait of a colossal dragon with obsidian scales, glowing amber eyes, smoke curling "
     "from flared nostrils, massive curved horns, dramatic stormy sky behind it, high fantasy, "
     "photorealistic"),
    ("WIZARD",
     "Portrait of an elderly wizard in deep blue robes covered with silver stars, casting a "
     "glowing spell, ancient books floating around him, magical energy crackling from his staff, "
     "high fantasy, photorealistic"),
    ("GOBLIN",
     "Portrait of a sneaky green-skinned goblin with large pointed ears, crooked yellow teeth, "
     "wide glinting eyes, clutching a stolen jewel, torchlit cave background, high fantasy, "
     "photorealistic"),
    ("KNIGHT",
     "Portrait of a noble knight in gleaming silver full plate armour, ornate helmet under one "
     "arm, determined expression, castle courtyard background, high fantasy, photorealistic"),
    ("ELF",
     "Portrait of a wise elven warrior with pointed ears, silver hair, piercing blue eyes, "
     "wearing intricate golden leaf armour, ethereal forest background, high fantasy, "
     "photorealistic"),
    ("DWARF",
     "Portrait of a stout dwarf warrior with a long braided red beard adorned with golden "
     "rings, a battle axe over one shoulder, runic engraved armour, high fantasy, "
     "photorealistic"),
    ("WITCH",
     "Portrait of an old witch with sharp green eyes, long silver hair, a wide-brimmed hat "
     "decorated with crow feathers and dried herbs, a knowing smile, candlelit room, high "
     "fantasy, photorealistic"),
    ("GOLEM",
     "Portrait of a massive stone golem with glowing orange eyes, carved arcane runes across "
     "its rocky face, cracked granite skin, looming and ancient, high fantasy, photorealistic"),
    ("VALKYRIE",
     "Portrait of a fierce valkyrie in silver winged armour, long golden hair streaming in the "
     "wind, a glowing spear raised high, dramatic storm clouds behind her, high fantasy, "
     "photorealistic"),
    ("SORCERER",
     "Portrait of a gaunt dark sorcerer in flowing black and purple robes, glowing violet eyes, "
     "a skull-topped staff, tendrils of dark energy swirling from his hands, high fantasy, "
     "photorealistic"),
    ("TROLL",
     "Portrait of a massive cave troll with jagged uneven teeth, a flat wide nose, small yellow "
     "eyes, warty grey-green skin, clutching a crude stone club, high fantasy, photorealistic"),
    # creatures — close up ────────────────────────────────────────────────────
    ("HYDRA",
     "Close up of a fearsome hydra with three serpent heads rearing up from dark swamp water, "
     "dripping scales, forked tongues, glowing red eyes, high fantasy, photorealistic"),
    ("PEGASUS",
     "Close up of a majestic white pegasus rearing up, enormous feathered wings spread wide, "
     "golden light on its silver mane, dramatic storm clouds, high fantasy, photorealistic"),
    ("PHOENIX",
     "Close up of a radiant phoenix rising from golden flames, crimson and gold feathers "
     "blazing, fierce amber eyes, trailing embers against a dark sky, high fantasy, "
     "photorealistic"),
    ("UNICORN",
     "Close up of a graceful unicorn with a pure white coat, spiraling silver horn glowing "
     "with magic, flowing silver mane, surrounded by ethereal forest light, high fantasy, "
     "photorealistic"),
]

WORDS_TO_WIN   = 10    # correct guesses needed to win (out of 30 available)
INPUT_FIELD_LEN = 10   # fixed width of the underscore input display

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 768            # postprocessed image size (square)
DISPLAY_HEIGHT = 512            # visible viewport height (centre-cropped vertically)
DITHER_SIZE    = 384            # intermediate size for dithering
CAMERA_STEP    = 0.05           # world-units per key press
CAMERA_TURN_ANGLE = np.radians(0.5)  # radians per rotation press
MAX_YAW        = np.radians(5)  # maximum yaw from origin
MAX_PITCH      = np.radians(5)  # maximum pitch from origin
MAX_Z          = 0.5            # maximum forward/backward from origin

MIRO_REWARDS = {"hpsv2_score": 0.75, "vqa_score": 0.5, "sciscore_score": 0.5}

BAYER_4x4 = np.array(
    [
        [0,  8,  2, 10],
        [12,  4, 14,  6],
        [3, 11,  1,  9],
        [15,  7, 13,  5],
    ],
    dtype=np.float32,
) / 16.0


# ── Image utilities ───────────────────────────────────────────────────────────
def dither(img: Image.Image, r: int = 4, g: int = 4, b: int = 4) -> Image.Image:
    src = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w, _ = src.shape
    bayer = np.tile(BAYER_4x4, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
    out = np.empty_like(src)
    for i, levels in enumerate([r, g, b]):
        step = 1.0 / (levels - 1)
        shifted = np.clip(src[:, :, i] + (bayer - 0.5) * step, 0.0, 1.0)
        out[:, :, i] = np.round(shifted * (levels - 1)) / (levels - 1)
    return Image.fromarray((out * 255).round().astype(np.uint8), mode="RGB")


def postprocess(img: Image.Image, r: int = 4, g: int = 4, b: int = 4) -> Image.Image:
    img = img.resize((DITHER_SIZE, DITHER_SIZE), Image.Resampling.BICUBIC)
    img = dither(img, r, g, b)
    img = img.resize((WINDOW_SIZE, WINDOW_SIZE), Image.Resampling.NEAREST)
    return img


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.dim() == 4:
        t = t.squeeze(0)
    arr = t.permute(1, 2, 0).cpu().float().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_surface(img: Image.Image) -> pygame.Surface:
    rgb = img.convert("RGB")
    return pygame.image.fromstring(rgb.tobytes(), rgb.size, "RGB")


# ── Camera helpers ────────────────────────────────────────────────────────────
_NAV_KEYS = (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN,
             pygame.K_z, pygame.K_s)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def make_cam_token(cam_rot: np.ndarray, cam_trans: np.ndarray,
                   image_size: int, device: torch.device) -> torch.Tensor:
    ext_np = np.concatenate([cam_rot, cam_trans.reshape(3, 1)], axis=1).astype(np.float32)
    ext = torch.from_numpy(ext_np).unsqueeze(0).unsqueeze(0).to(device)
    dummy_intr = torch.zeros(1, 1, 3, 3, device=device)
    camera = extri_intri_to_pose_encoding(
        extrinsics=ext,
        intrinsics=dummy_intr,
        image_size_hw=(image_size, image_size),
    )
    return camera[..., :7].view(1, 7)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="High Fantasy Word Explorer")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance", type=float, default=7.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required.")
    device = torch.device("cuda")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading MIRO …")
    miro = MiroPipeline.from_pretrained("nicolas-dufour/miro").to(device, torch.bfloat16)

    print("Loading OVIE …")
    ovie: OVIEModel = OVIEModel.from_pretrained("kyutai/ovie", revision="v1.0").to(device)
    ovie.eval()
    ovie_size: int = ovie.image_size

    # Single generator; each call advances its state so every scene is different.
    random.seed(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed)

    # ── Game state ─────────────────────────────────────────────────────────────
    scene_order = list(range(len(WORDS_AND_PROMPTS)))
    random.shuffle(scene_order)
    current_idx   = 0          # index into scene_order
    found_words: list[str] = []
    lives     = 3
    game_over = False
    text_mode  = False
    typed_text = ""
    show_help  = False

    def scene_word()   -> str: return WORDS_AND_PROMPTS[scene_order[current_idx]][0]
    def scene_prompt() -> str: return WORDS_AND_PROMPTS[scene_order[current_idx]][1]

    # ── Camera state ───────────────────────────────────────────────────────────
    cam_rot   = np.eye(3, dtype=np.float64)
    cam_trans = np.zeros(3, dtype=np.float64)
    total_yaw   = 0.0
    total_pitch = 0.0

    lock      = threading.Lock()
    generating = threading.Event()
    held_key: list[int | None] = [None]

    # Shared mutable references updated by workers
    original_raw:     list[Image.Image] = [None]  # type: ignore[list-item]
    original_display: list[Image.Image] = [None]  # type: ignore[list-item]
    current_display:  list[Image.Image] = [None]  # type: ignore[list-item]

    def _reset_camera() -> None:
        cam_rot[:]   = np.eye(3)
        cam_trans[:] = 0.0
        nonlocal total_yaw, total_pitch
        total_yaw   = 0.0
        total_pitch = 0.0

    # ── MIRO worker ────────────────────────────────────────────────────────────
    def load_scene(prompt: str) -> None:
        nonlocal total_yaw, total_pitch

        print(f"Generating scene {current_idx + 1}/{len(WORDS_AND_PROMPTS)} …")
        with torch.inference_mode():
            images = miro(
                prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=1,
                reward_targets=MIRO_REWARDS,
                generator=generator,
            )
        new_image = images[0]
        new_raw   = new_image.resize((ovie_size, ovie_size), Image.Resampling.BICUBIC)
        new_disp  = postprocess(new_image)

        with lock:
            original_raw[0]     = new_raw
            original_display[0] = new_disp
            current_display[0]  = new_disp
            _reset_camera()

        generating.clear()

    # ── OVIE worker ────────────────────────────────────────────────────────────
    def move(initial_key: int) -> None:
        nonlocal total_yaw, total_pitch

        key = initial_key
        while True:
            new_rot   = cam_rot.copy()
            new_trans = cam_trans.copy()
            new_yaw   = total_yaw
            new_pitch = total_pitch

            if key == pygame.K_z:
                candidate = cam_trans[2] + CAMERA_STEP
                if abs(candidate) > MAX_Z:
                    break
                new_trans[2] = candidate
            elif key == pygame.K_s:
                candidate = cam_trans[2] - CAMERA_STEP
                if abs(candidate) > MAX_Z:
                    break
                new_trans[2] = candidate
            elif key == pygame.K_LEFT:
                candidate = total_yaw - CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_YAW:
                    break
                R = _rot_y(-CAMERA_TURN_ANGLE)
                new_rot, new_trans, new_yaw = R @ cam_rot, R @ cam_trans, candidate
            elif key == pygame.K_RIGHT:
                candidate = total_yaw + CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_YAW:
                    break
                R = _rot_y(CAMERA_TURN_ANGLE)
                new_rot, new_trans, new_yaw = R @ cam_rot, R @ cam_trans, candidate
            elif key == pygame.K_UP:
                candidate = total_pitch - CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_PITCH:
                    break
                R = _rot_x(-CAMERA_TURN_ANGLE)
                new_rot, new_trans, new_pitch = R @ cam_rot, R @ cam_trans, candidate
            elif key == pygame.K_DOWN:
                candidate = total_pitch + CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_PITCH:
                    break
                R = _rot_x(CAMERA_TURN_ANGLE)
                new_rot, new_trans, new_pitch = R @ cam_rot, R @ cam_trans, candidate

            with torch.inference_mode():
                img_t = ToTensor()(original_raw[0]).unsqueeze(0).to(device)
                cam_t = make_cam_token(new_rot, new_trans, ovie_size, device)
                pred  = ovie(x=img_t, cam_params=cam_t)

            with lock:
                current_display[0] = postprocess(tensor_to_pil(pred))
                cam_rot[:]   = new_rot
                cam_trans[:] = new_trans
                total_yaw    = new_yaw
                total_pitch  = new_pitch

            next_key = held_key[0]
            if next_key is None:
                break
            key = next_key

        generating.clear()

    # ── Generate first scene ───────────────────────────────────────────────────
    print(f'Scene 1/{len(WORDS_AND_PROMPTS)}: generating …')
    generating.set()
    load_scene(scene_prompt())   # runs synchronously before the window opens

    # ── Pygame setup ───────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, DISPLAY_HEIGHT))
    pygame.display.set_caption("High Fantasy Word Explorer")
    _crop_y = -(WINDOW_SIZE - DISPLAY_HEIGHT) // 2

    font       = pygame.font.SysFont("monospace", 14)
    font_large = pygame.font.SysFont("monospace", 32, bold=True)

    clock = pygame.time.Clock()
    running = True

    while running:
        # ── Events ─────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                # ── Text-input mode ───────────────────────────────────────────
                if text_mode:
                    if event.key in (pygame.K_ESCAPE, pygame.K_TAB):
                        text_mode  = False
                        typed_text = ""

                    elif event.key == pygame.K_RETURN and not generating.is_set():
                        guess = typed_text.strip().upper()
                        text_mode  = False
                        typed_text = ""

                        if guess == scene_word():
                            found_words.append(scene_word())
                            if len(found_words) < WORDS_TO_WIN and current_idx < len(WORDS_AND_PROMPTS) - 1:
                                current_idx += 1
                                generating.set()
                                threading.Thread(
                                    target=load_scene, args=(scene_prompt(),), daemon=True
                                ).start()
                            # else: all words found, nothing to generate
                        else:
                            lives -= 1
                            if lives <= 0:
                                game_over = True
                            else:
                                generating.set()
                                threading.Thread(
                                    target=load_scene, args=(scene_prompt(),), daemon=True
                                ).start()

                    elif event.key == pygame.K_BACKSPACE:
                        typed_text = typed_text[:-1]

                    elif event.unicode and event.unicode.isalpha():
                        if len(typed_text) < INPUT_FIELD_LEN:
                            typed_text += event.unicode.upper()

                # ── Navigation mode ───────────────────────────────────────────
                else:
                    if event.key in (pygame.K_q,):
                        running = False
                    elif event.key == pygame.K_ESCAPE:
                        if show_help:
                            show_help = False
                        else:
                            running = False
                    elif event.unicode == "?":
                        show_help = not show_help
                    elif event.key == pygame.K_RETURN and not generating.is_set() and \
                            (game_over or len(found_words) >= WORDS_TO_WIN):
                        # Restart — Python random state already advanced, so shuffle differs
                        random.shuffle(scene_order)
                        current_idx = 0
                        found_words.clear()
                        lives     = 3
                        game_over = False
                        show_help = False
                        with lock:
                            _reset_camera()
                        generating.set()
                        threading.Thread(
                            target=load_scene, args=(scene_prompt(),), daemon=True
                        ).start()
                    elif event.key == pygame.K_TAB and not generating.is_set() \
                            and not game_over and not show_help:
                        text_mode  = True
                        typed_text = ""
                    elif event.key == pygame.K_r and not generating.is_set():
                        with lock:
                            current_display[0] = original_display[0]
                            _reset_camera()
                    elif event.key in _NAV_KEYS and not show_help:
                        held_key[0] = event.key
                        if not generating.is_set():
                            generating.set()
                            threading.Thread(
                                target=move, args=(event.key,), daemon=True
                            ).start()

            elif event.type == pygame.KEYUP:
                if event.key in _NAV_KEYS and held_key[0] == event.key:
                    held_key[0] = None

        # ── Render ─────────────────────────────────────────────────────────────
        with lock:
            display_img = current_display[0]

        screen.blit(pil_to_surface(display_img), (0, _crop_y))

        # Status bar (top-left)
        if generating.is_set():
            lbl = font.render(" Generating… ", True, (220, 220, 220), (0, 0, 0))
        else:
            yaw_s   = f"{np.degrees(total_yaw):+.0f}°"
            pitch_s = f"{np.degrees(total_pitch):+.0f}°"
            lbl = font.render(f" yaw {yaw_s}  pitch {pitch_s}  z {cam_trans[2]:+.2f} ",
                              True, (160, 160, 160), (0, 0, 0))
        screen.blit(lbl, (8, 8))

        # Lives + found-words panel (top-right)
        n_found = len(found_words)

        # Draw lives as circles (no font rendering, works on any system)
        _LR, _LG = 10, 8           # circle radius, gap between circles
        _lx0 = WINDOW_SIZE - 8 - 3 * (2 * _LR) - 2 * _LG + _LR
        _lcy = 8 + _LR
        for i in range(3):
            cx = _lx0 + i * (2 * _LR + _LG)
            pygame.draw.circle(screen, (0, 0, 0),       (cx, _lcy), _LR + 1)
            if i < lives:
                pygame.draw.circle(screen, (210, 35, 35), (cx, _lcy), _LR)
            else:
                pygame.draw.circle(screen, (55, 15, 15), (cx, _lcy), _LR)

        fy = _lcy + _LR + 6
        header = font.render(f" FOUND {n_found}/{WORDS_TO_WIN} ", True, (200, 200, 200), (0, 0, 0))
        screen.blit(header, (WINDOW_SIZE - header.get_width() - 8, fy))
        fy += header.get_height() + 2
        for w in found_words:
            ws = font.render(f" {w} ", True, (100, 255, 100), (0, 0, 0))
            screen.blit(ws, (WINDOW_SIZE - ws.get_width() - 8, fy))
            fy += ws.get_height() + 2

        # Win / game-over overlays
        if n_found >= WORDS_TO_WIN or game_over:
            if n_found >= WORDS_TO_WIN:
                big_surf = font_large.render("  YOU WIN!  ", True, (255, 215, 0), (0, 0, 0))
            else:
                big_surf = font_large.render("  YOU DIED  ", True, (200, 0, 0), (0, 0, 0))
            cy = (DISPLAY_HEIGHT - big_surf.get_height()) // 2
            screen.blit(big_surf, ((WINDOW_SIZE - big_surf.get_width()) // 2, cy))
            restart_hint = font.render(" ENTER to play again ",
                                       True, (180, 180, 180), (0, 0, 0))
            screen.blit(restart_hint,
                        ((WINDOW_SIZE - restart_hint.get_width()) // 2,
                         cy + big_surf.get_height() + 4))

        # Text-input overlay
        if text_mode:
            BOX_W, BOX_H = 400, 90
            box_x = (WINDOW_SIZE - BOX_W) // 2
            box_y = (DISPLAY_HEIGHT - BOX_H) // 2

            overlay = pygame.Surface((BOX_W, BOX_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 210))
            screen.blit(overlay, (box_x, box_y))

            # Underscore-filled input field: typed chars + remaining underscores
            field = (typed_text + "_" * (INPUT_FIELD_LEN - len(typed_text)))
            field_surf = font_large.render(field, True, (255, 255, 255))
            screen.blit(field_surf,
                        (box_x + (BOX_W - field_surf.get_width()) // 2,
                         box_y + (BOX_H - field_surf.get_height()) // 2))

            hint = font.render(" ENTER to guess · ESC to cancel ",
                               True, (160, 160, 160))
            screen.blit(hint, (box_x + (BOX_W - hint.get_width()) // 2,
                                box_y + BOX_H - hint.get_height() - 4))

        # "? help" prompt (bottom-left)
        help_lbl = font.render(" ? help ", True, (110, 110, 110))
        screen.blit(help_lbl, (8, DISPLAY_HEIGHT - help_lbl.get_height() - 6))

        # Help overlay
        if show_help:
            LINES = [
                "  CONTROLS",
                "",
                "  Z / S         move forward / backward",
                "  Left / Right  turn left / right",
                "  Up / Down     look up / down",
                "  R             reset camera",
                "",
                "  Tab           open word-input field",
                "  Enter         submit guess",
                "  Backspace     delete character",
                "",
                "  ?             show / hide help",
                "  ESC / Q       quit",
            ]
            lh   = font.get_linesize()
            pad  = 16
            bw   = 420
            bh   = len(LINES) * lh + 2 * pad
            bx   = (WINDOW_SIZE - bw) // 2
            by   = (DISPLAY_HEIGHT - bh) // 2
            ov   = pygame.Surface((bw, bh), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 220))
            screen.blit(ov, (bx, by))
            for k, line in enumerate(LINES):
                col = (255, 215, 0) if k == 0 else (200, 200, 200)
                screen.blit(font.render(line, True, col),
                            (bx + pad, by + pad + k * lh))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
