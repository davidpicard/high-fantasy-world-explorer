#!/usr/bin/env python3
"""
High Fantasy Word Explorer

Explore AI-generated scenes and guess the word each one depicts.
Press Tab to open the input field, type your answer, press Enter.
Find 10 words to win!

Usage:
    python game.py
    OVIE_PATH=/path/to/ovie python game.py

Controls:
    Z / S        – move forward / backward
    Left / Right – yaw (full 360°)
    Up / Down    – pitch (look up / down)
    Tab          – open / close word-input field
    Enter        – submit guess
    Backspace    – delete last character
    R            – reset camera to origin
    ESC / Q      – close overlay / quit
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

# ── Word list ─────────────────────────────────────────────────────────────────
# Each entry: (word, north_prompt, background_prompt)
# north_prompt  – scene containing the concept to guess (shown facing north)
# background_prompt – coherent skybox for east / south / west views (no spoiler)
WORDS_AND_PROMPTS = [
    # scenes ──────────────────────────────────────────────────────────────────
    ("CASTLE",
     "A grand medieval stone castle rising from a misty sea cliff at golden hour, tall towers "
     "with glowing windows, a drawbridge over a dark moat, ravens circling, high fantasy, "
     "photorealistic",
     "Rocky coastal cliffs and mist-shrouded sea at golden hour, crashing waves on dark rocks, "
     "no buildings, high fantasy, photorealistic"),

    ("FOREST",
     "An ancient enchanted forest with enormous silver-barked trees, bioluminescent mushrooms "
     "and glowing flowers on the mossy floor, magical golden light rays filtering through the "
     "dense canopy, high fantasy, photorealistic",
     "Ancient silver-barked trees and glowing mossy undergrowth in an enchanted forest, magical "
     "light filtering through dense canopy, high fantasy, photorealistic"),

    ("DUNGEON",
     "A dark underground dungeon with rough stone walls and iron-barred cells, flickering "
     "torchlight casting long shadows on bones and rusted weapons on the damp floor, heavy "
     "chains on the walls, gothic high fantasy, photorealistic",
     "Dark stone underground corridors with iron-barred archways, flickering wall torches and "
     "damp rough-hewn walls, gothic high fantasy, photorealistic"),

    ("TAVERN",
     "Interior of a warm medieval fantasy tavern, low wooden beams, a large stone fireplace "
     "roaring with fire, cloaked adventurers at oak tables with tankards of ale, warm "
     "candlelight, high fantasy, photorealistic",
     "Warm candlelit medieval interior with low wooden beams, stone walls hung with lanterns, "
     "empty wooden tables and benches, high fantasy, photorealistic"),

    ("PORTAL",
     "A swirling circular arcane portal of violet and gold energy suspended between ancient "
     "moss-covered stone pillars in a ruin, glimpses of another realm through the gateway, "
     "glowing runes carved into the stone, high fantasy, photorealistic",
     "Ancient moss-covered stone ruins and pillars in a misty forest, crumbling archways and "
     "carved stones, no portal, high fantasy, photorealistic"),

    ("THRONE",
     "An imposing dark throne room with a massive obsidian throne on a raised dais, towering "
     "pillars lined with burning braziers, tattered battle banners hanging from vaulted "
     "ceilings, dramatic candlelight, high fantasy, photorealistic",
     "Dark stone hall with towering pillars, burning braziers and vaulted stone ceilings, "
     "tattered banners, no throne, high fantasy, photorealistic"),

    ("RUINS",
     "Ancient stone ruins of a fallen elven city overgrown with vines and glowing moss, "
     "crumbling archways and broken statues, mist drifting through, high fantasy, "
     "photorealistic",
     "Dense misty ancient forest with crumbling stone archways and fallen carved columns "
     "overgrown with vines and glowing moss, high fantasy, photorealistic"),

    ("CRYPT",
     "A vast underground crypt with rows of stone sarcophagi carved with warrior reliefs, "
     "flickering torch sconces on damp walls, cobwebs and scattered bones, high fantasy, "
     "photorealistic",
     "Dark underground stone halls with carved arched ceilings, torch sconces, cobwebs and "
     "damp stone walls, high fantasy, photorealistic"),

    ("SHRINE",
     "A moss-covered outdoor forest shrine with a stone idol surrounded by offerings of "
     "candles and flowers, shafts of magical dappled light, ancient mystical atmosphere, "
     "high fantasy, photorealistic",
     "Magical mossy forest glade with dappled golden light through ancient trees, scattered "
     "wildflowers and glowing mushrooms, high fantasy, photorealistic"),

    ("ALTAR",
     "A dark stone altar in an underground ritual chamber, carved with arcane symbols, "
     "surrounded by burning black candles, ominous light from above, high fantasy, "
     "photorealistic",
     "Dark underground ritual chamber with rough stone walls, burning black candles and arcane "
     "carvings, no altar, high fantasy, photorealistic"),

    ("FORGE",
     "Interior of a dwarven forge, a massive furnace blazing with orange fire, sparks flying, "
     "glowing hot iron on a huge anvil, weapons and tools hung on stone walls, high fantasy, "
     "photorealistic",
     "Dwarven stone cavern with rough rock walls, glowing coals and metal tools on stone walls, "
     "no forge or anvil, high fantasy, photorealistic"),

    # objects — close up ───────────────────────────────────────────────────────
    ("POTION",
     "Close up of a cluttered alchemist workshop with glowing coloured potions in glass vials "
     "and flasks, a bubbling cauldron emitting rainbow smoke, dried herbs, an open ancient "
     "grimoire, candlelight, high fantasy, photorealistic",
     "Cluttered alchemist workshop shelves with glass bottles, dried herbs and dusty tomes, "
     "candlelight, no glowing potions, high fantasy, photorealistic"),

    ("CROWN",
     "Close up of an ancient royal crown wrought from dark twisted gold, set with glowing "
     "rubies and sapphires, resting on red velvet, dramatic side lighting, high fantasy, "
     "photorealistic",
     "Dark stone treasury interior with velvet pedestals and dim candlelight, stone walls with "
     "carved reliefs, no crown, high fantasy, photorealistic"),

    ("GRIMOIRE",
     "Close up of an open ancient spellbook with yellowed pages covered in glowing arcane "
     "symbols and diagrams, a quill pen resting on the page, flickering candlelight, high "
     "fantasy, photorealistic",
     "Ancient magical library with wooden shelves of dusty tomes and scrolls, candlelight and "
     "arcane instruments, no open book, high fantasy, photorealistic"),

    ("SWORD",
     "Close up of a legendary enchanted sword with a jewelled crossguard, glowing runes etched "
     "along the gleaming blade, embedded in a mossy stone, dramatic lighting, high fantasy, "
     "photorealistic",
     "Ancient mossy stone ruins in a dramatic forest glade, dappled light and ferns, no sword, "
     "high fantasy, photorealistic"),

    # persons and creatures — portrait ────────────────────────────────────────
    ("DRAGON",
     "Portrait of a colossal dragon with obsidian scales, glowing amber eyes, smoke curling "
     "from flared nostrils, massive curved horns, dramatic stormy sky behind it, high fantasy, "
     "photorealistic",
     "Dark stormy sky with billowing storm clouds and lightning over rocky mountain peaks, "
     "high fantasy, photorealistic"),

    ("WIZARD",
     "Portrait of an elderly wizard in deep blue robes covered with silver stars, casting a "
     "glowing spell, ancient books floating around him, magical energy crackling from his staff, "
     "high fantasy, photorealistic",
     "Ancient magical study filled with floating books, arcane instruments and glowing orbs, "
     "no figure, high fantasy, photorealistic"),

    ("GOBLIN",
     "Portrait of a sneaky green-skinned goblin with large pointed ears, crooked yellow teeth, "
     "wide glinting eyes, clutching a stolen jewel, torchlit cave background, high fantasy, "
     "photorealistic",
     "Dark torchlit cave interior with rough stone walls and scattered treasures, no goblin, "
     "high fantasy, photorealistic"),

    ("KNIGHT",
     "Portrait of a noble knight in gleaming silver full plate armour, ornate helmet under one "
     "arm, determined expression, castle courtyard background, high fantasy, photorealistic",
     "Stone castle courtyard with flagstone floor, stone walls hung with banners, afternoon "
     "light, no figure, high fantasy, photorealistic"),

    ("ELF",
     "Portrait of a wise elven warrior with pointed ears, silver hair, piercing blue eyes, "
     "wearing intricate golden leaf armour, ethereal forest background, high fantasy, "
     "photorealistic",
     "Ethereal ancient forest with silver-barked trees and magical golden light, no figure, "
     "high fantasy, photorealistic"),

    ("DWARF",
     "Portrait of a stout dwarf warrior with a long braided red beard adorned with golden "
     "rings, a battle axe over one shoulder, runic engraved armour, high fantasy, "
     "photorealistic",
     "Vast underground dwarven hall with stone pillars and distant forge fires, no figure, "
     "high fantasy, photorealistic"),

    ("WITCH",
     "Portrait of an old witch with sharp green eyes, long silver hair, a wide-brimmed hat "
     "decorated with crow feathers and dried herbs, a knowing smile, candlelit room, high "
     "fantasy, photorealistic",
     "Candlelit cottage interior with dried herbs hanging from rafters and shelves of jars, "
     "no figure, high fantasy, photorealistic"),

    ("GOLEM",
     "Portrait of a massive stone golem with glowing orange eyes, carved arcane runes across "
     "its rocky face, cracked granite skin, looming and ancient, high fantasy, photorealistic",
     "Ancient stone chamber with massive carved walls and glowing arcane runes, no golem, "
     "high fantasy, photorealistic"),

    ("VALKYRIE",
     "Portrait of a fierce valkyrie in silver winged armour, long golden hair streaming in the "
     "wind, a glowing spear raised high, dramatic storm clouds behind her, high fantasy, "
     "photorealistic",
     "Dramatic storm clouds with lightning and rays of golden light over mountainous landscape, "
     "high fantasy, photorealistic"),

    ("SORCERER",
     "Portrait of a gaunt dark sorcerer in flowing black and purple robes, glowing violet eyes, "
     "a skull-topped staff, tendrils of dark energy swirling from his hands, high fantasy, "
     "photorealistic",
     "Dark stone tower interior with arcane symbols carved on walls, purple energy crackling "
     "in the air, no figure, high fantasy, photorealistic"),

    ("TROLL",
     "Portrait of a massive cave troll with jagged uneven teeth, a flat wide nose, small yellow "
     "eyes, warty grey-green skin, clutching a crude stone club, high fantasy, photorealistic",
     "Dark rocky cave with jagged stone walls and dim greenish bioluminescent light, no troll, "
     "high fantasy, photorealistic"),

    # creatures — close up ────────────────────────────────────────────────────
    ("HYDRA",
     "Close up of a fearsome hydra with three serpent heads rearing up from dark swamp water, "
     "dripping scales, forked tongues, glowing red eyes, high fantasy, photorealistic",
     "Dark murky swamp with gnarled dead trees and fog drifting over still black water, "
     "high fantasy, photorealistic"),

    ("PEGASUS",
     "Close up of a majestic white pegasus rearing up, enormous feathered wings spread wide, "
     "golden light on its silver mane, dramatic storm clouds, high fantasy, photorealistic",
     "Dramatic storm clouds with shafts of golden light over mountainous landscape, "
     "high fantasy, photorealistic"),

    ("PHOENIX",
     "Close up of a radiant phoenix rising from golden flames, crimson and gold feathers "
     "blazing, fierce amber eyes, trailing embers against a dark sky, high fantasy, "
     "photorealistic",
     "Dark twilight sky with glowing embers and smoke clouds, volcanic rocky landscape below, "
     "high fantasy, photorealistic"),

    ("UNICORN",
     "Close up of a graceful unicorn with a pure white coat, spiraling silver horn glowing "
     "with magic, flowing silver mane, surrounded by ethereal forest light, high fantasy, "
     "photorealistic",
     "Magical misty forest with silver light filtering through ancient trees, wildflowers and "
     "glowing moss, high fantasy, photorealistic"),
]

WORDS_TO_WIN    = 10
INPUT_FIELD_LEN = 10

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 768
DISPLAY_HEIGHT = 512
DITHER_SIZE    = 384

# Navigation step sizes (same as original fine-grained controls)
CAMERA_STEP       = 0.05            # Z translation per key press
CAMERA_TURN_ANGLE = np.radians(1) # yaw / pitch step (radians)
MAX_PITCH         = np.radians(5)   # ±5° pitch limit
MAX_Z             = 0.5             # ±0.5 Z limit
# 4 MIRO references at 0°/90°/180°/270° (N/E/S/W).
# OVIE chaining at CHAIN_STEP_DEG increments; transitions fade to black at the
# midpoint between references so OVIE quality degradation is masked by darkness.
NUM_REFS        = 4
SECTOR_DEG      = 360.0 / NUM_REFS   # 90°
CHAIN_STEP_DEG  = 2.0
MAX_CHAIN_STEPS = 16    # 16 × 2° = 32° reliable range per reference

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


def postprocess(img: Image.Image, r: int = 8, g: int = 8, b: int = 8) -> Image.Image:
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


# ── Camera helpers ─────────────────────────────────────────────────────────────
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
    parser.add_argument("--seed",     type=int,   default=3407)
    parser.add_argument("--steps",    type=int,   default=40)
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

    random.seed(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed)

    # ── Game state ─────────────────────────────────────────────────────────────
    scene_order = list(range(len(WORDS_AND_PROMPTS)))
    random.shuffle(scene_order)
    current_idx   = 0
    found_words: list[str] = []
    lives      = 3
    game_over  = False
    text_mode  = False
    typed_text = ""
    show_help  = False

    def scene_word()        -> str: return WORDS_AND_PROMPTS[scene_order[current_idx]][0]
    def scene_word_prompt() -> str: return WORDS_AND_PROMPTS[scene_order[current_idx]][1]
    def scene_bg_prompt()   -> str: return WORDS_AND_PROMPTS[scene_order[current_idx]][2]

    # ── Navigation state ───────────────────────────────────────────────────────
    # total_yaw in degrees [0, 360); 4 reference images at 0°/90°/180°/270°.
    # ref 0 = N (word prompt); refs 1-3 = E/S/W (bg prompt).
    # ovie_cache[ref][delta_int] = cached PIL at 3°-grid multiples from that reference.
    total_yaw   = 0.0           # degrees
    total_pitch = 0.0           # radians, ±MAX_PITCH
    cam_z       = 0.0           # ±MAX_Z

    lock       = threading.Lock()
    generating = threading.Event()
    held_key: list[int | None] = [None]

    ref_raws:   list[Image.Image | None] = [None] * NUM_REFS
    ovie_cache: list[dict[int, Image.Image]] = [{} for _ in range(NUM_REFS)]
    north_display:   list[Image.Image | None] = [None]
    current_display: list[Image.Image | None] = [None]

    def _reset_camera(yaw: float = 0.0) -> None:
        nonlocal total_yaw, total_pitch, cam_z
        total_yaw   = yaw
        total_pitch = 0.0
        cam_z       = 0.0

    # ── OVIE helpers ───────────────────────────────────────────────────────────
    def _ovie_step(img: Image.Image, delta_yaw_deg: float) -> Image.Image:
        """One OVIE inference with a relative yaw of delta_yaw_deg from img's viewpoint."""
        rot   = _rot_y(np.radians(delta_yaw_deg))
        cam_t = make_cam_token(rot, np.zeros(3), ovie_size, device)
        img_t = ToTensor()(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            pred = ovie(x=img_t, cam_params=cam_t)
        return tensor_to_pil(pred)

    def _ensure_grid(ref_idx: int, target_deg: float) -> tuple[Image.Image, float]:
        """
        Build the cached OVIE chain for ref_idx up to the grid point at or
        below |target_deg|.  Returns (grid_image, remaining_deg) where
        |remaining_deg| < CHAIN_STEP_DEG.  The remaining sub-grid angle is
        left to the caller so it can be composed with other transforms.
        """
        sign     = 1 if target_deg >= 0 else -1
        n_links  = int(abs(target_deg) / CHAIN_STEP_DEG)
        grid_int = int(n_links * CHAIN_STEP_DEG) * sign
        step     = int(CHAIN_STEP_DEG) * sign
        prev     = 0
        for _ in range(n_links):
            cur = prev + step
            if cur not in ovie_cache[ref_idx]:
                ovie_cache[ref_idx][cur] = _ovie_step(ovie_cache[ref_idx][prev],
                                                       float(step))
            prev = cur
        grid_img = ovie_cache[ref_idx].get(grid_int, ref_raws[ref_idx])
        return grid_img, target_deg - grid_int

    def get_full_view(yaw_deg: float, pitch_rad: float, z: float) -> Image.Image:
        """
        Return a postprocessed display image for the given camera state.

        Yaw is handled by the cached OVIE chain (grid at CHAIN_STEP_DEG
        multiples).  The sub-grid yaw remainder, pitch, and Z are all folded
        into a single final OVIE call so that non-zero pitch/Z never costs an
        extra inference on top of the yaw chain — at most one uncached call
        per frame regardless of camera state.
        """
        theta  = yaw_deg % 360.0
        sector = int(theta / SECTOR_DEG) % NUM_REFS
        t      = (theta % SECTOR_DEG) / SECTOR_DEG

        if t <= 0.5:
            ref_idx = sector
            delta   = t * SECTOR_DEG
        else:
            ref_idx = (sector + 1) % NUM_REFS
            delta   = (t - 1.0) * SECTOR_DEG

        d          = abs(delta) / (SECTOR_DEG / 2.0)
        brightness = np.cos(d * (np.pi / 2.0)) ** 2

        if brightness < 0.01:
            return postprocess(Image.new("RGB", (ovie_size, ovie_size), (0, 0, 0)))

        grid_img, remaining_yaw = _ensure_grid(ref_idx, delta)

        # Compose sub-grid yaw + pitch + Z into one OVIE call
        if abs(remaining_yaw) > 0.01 or abs(pitch_rad) > 1e-6 or abs(z) > 1e-6:
            rot   = _rot_x(pitch_rad) @ _rot_y(np.radians(remaining_yaw))
            trans = np.array([0.0, 0.0, z])
            cam_t = make_cam_token(rot, trans, ovie_size, device)
            img_t = ToTensor()(grid_img).unsqueeze(0).to(device)
            with torch.inference_mode():
                pred = ovie(x=img_t, cam_params=cam_t)
            raw = tensor_to_pil(pred)
        else:
            raw = grid_img

        if brightness < 0.999:
            arr = np.array(raw, dtype=np.float32) * brightness
            raw = Image.fromarray(arr.round().astype(np.uint8), mode="RGB")

        return postprocess(raw)

    # ── MIRO worker ────────────────────────────────────────────────────────────
    def load_scene(word_prompt: str, bg_prompt: str) -> None:
        nonlocal total_yaw, total_pitch, cam_z
        print(f"Generating scene {current_idx + 1}/{len(WORDS_AND_PROMPTS)} …")
        def _resize(img: Image.Image) -> Image.Image:
            return img.resize((ovie_size, ovie_size), Image.Resampling.BICUBIC)

        with torch.inference_mode():
            north_imgs = miro(
                word_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=1,
                reward_targets=MIRO_REWARDS,
                generator=generator,
            )
            bg_imgs = miro(
                bg_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                num_images_per_prompt=3,
                reward_targets=MIRO_REWARDS,
                generator=generator,
            )

        # refs: 0=N(word), 1=E, 2=S, 3=W
        new_raws = [
            _resize(north_imgs[0]),
            _resize(bg_imgs[0]),
            _resize(bg_imgs[1]),
            _resize(bg_imgs[2]),
        ]
        n_disp     = postprocess(north_imgs[0])
        start_ref  = random.randrange(NUM_REFS)
        start_disp = postprocess(new_raws[start_ref]) if start_ref != 0 else n_disp

        with lock:
            for i in range(NUM_REFS):
                ref_raws[i] = new_raws[i]
                ovie_cache[i].clear()
                ovie_cache[i][0] = new_raws[i]
            north_display[0]   = n_disp
            current_display[0] = start_disp
            _reset_camera(start_ref * SECTOR_DEG)

        generating.clear()

    # ── OVIE navigation worker ─────────────────────────────────────────────────
    def move(initial_key: int) -> None:
        nonlocal total_yaw, total_pitch, cam_z
        key = initial_key
        turn_deg = np.degrees(CAMERA_TURN_ANGLE)
        while True:
            new_yaw   = total_yaw
            new_pitch = total_pitch
            new_z     = cam_z

            if key == pygame.K_LEFT:
                new_yaw = (total_yaw - turn_deg) % 360.0
            elif key == pygame.K_RIGHT:
                new_yaw = (total_yaw + turn_deg) % 360.0
            elif key == pygame.K_UP:
                candidate = total_pitch - CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_PITCH:
                    break
                new_pitch = candidate
            elif key == pygame.K_DOWN:
                candidate = total_pitch + CAMERA_TURN_ANGLE
                if abs(candidate) > MAX_PITCH:
                    break
                new_pitch = candidate
            elif key == pygame.K_z:
                candidate = cam_z + CAMERA_STEP
                if abs(candidate) > MAX_Z:
                    break
                new_z = candidate
            elif key == pygame.K_s:
                candidate = cam_z - CAMERA_STEP
                if abs(candidate) > MAX_Z:
                    break
                new_z = candidate
            else:
                break

            new_display = get_full_view(new_yaw, new_pitch, new_z)

            with lock:
                current_display[0] = new_display
                total_yaw   = new_yaw
                total_pitch = new_pitch
                cam_z       = new_z

            next_key = held_key[0]
            if next_key is None:
                break
            key = next_key

        generating.clear()

    # ── Generate first scene (synchronous, before window opens) ───────────────
    print(f'Scene 1/{len(WORDS_AND_PROMPTS)}: generating …')
    generating.set()
    load_scene(scene_word_prompt(), scene_bg_prompt())

    # ── Pygame setup ───────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, DISPLAY_HEIGHT))
    pygame.display.set_caption("High Fantasy Word Explorer")
    _crop_y = -(WINDOW_SIZE - DISPLAY_HEIGHT) // 2

    font       = pygame.font.SysFont("monospace", 14)
    font_large = pygame.font.SysFont("monospace", 32, bold=True)

    clock   = pygame.time.Clock()
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
                        guess      = typed_text.strip().upper()
                        text_mode  = False
                        typed_text = ""

                        if guess == scene_word():
                            found_words.append(scene_word())
                            if len(found_words) < WORDS_TO_WIN and \
                                    current_idx < len(WORDS_AND_PROMPTS) - 1:
                                current_idx += 1
                                generating.set()
                                threading.Thread(
                                    target=load_scene,
                                    args=(scene_word_prompt(), scene_bg_prompt()),
                                    daemon=True,
                                ).start()
                        else:
                            lives -= 1
                            if lives <= 0:
                                game_over = True
                            else:
                                generating.set()
                                threading.Thread(
                                    target=load_scene,
                                    args=(scene_word_prompt(), scene_bg_prompt()),
                                    daemon=True,
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
                        random.shuffle(scene_order)
                        current_idx = 0
                        found_words.clear()
                        lives      = 3
                        game_over  = False
                        show_help  = False
                        with lock:
                            _reset_camera()
                        generating.set()
                        threading.Thread(
                            target=load_scene,
                            args=(scene_word_prompt(), scene_bg_prompt()),
                            daemon=True,
                        ).start()
                    elif event.key == pygame.K_TAB and not generating.is_set() \
                            and not game_over and not show_help:
                        text_mode  = True
                        typed_text = ""
                    elif event.key == pygame.K_r and not generating.is_set():
                        with lock:
                            current_display[0] = north_display[0]
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
            lbl = font.render(
                f" yaw {total_yaw:.0f}°  pitch {np.degrees(total_pitch):+.0f}°  z {cam_z:+.2f} ",
                True, (160, 160, 160), (0, 0, 0))
        screen.blit(lbl, (8, 8))

        # Lives + found-words panel (top-right)
        n_found = len(found_words)
        _LR, _LG = 10, 8
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
            field      = typed_text + "_" * (INPUT_FIELD_LEN - len(typed_text))
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
                "  Left / Right  yaw (full 360°)",
                "  Up / Down     pitch (look up / down)",
                "  R             reset camera",
                "",
                "  Tab           open word-input field",
                "  Enter         submit guess",
                "  Backspace     delete character",
                "",
                "  ?             show / hide help",
                "  ESC / Q       quit",
            ]
            lh  = font.get_linesize()
            pad = 16
            bw  = 380
            bh  = len(LINES) * lh + 2 * pad
            bx  = (WINDOW_SIZE - bw) // 2
            by  = (DISPLAY_HEIGHT - bh) // 2
            ov  = pygame.Surface((bw, bh), pygame.SRCALPHA)
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
