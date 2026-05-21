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
# north_prompt       – scene at natural viewing distance; subject embedded in a
#                      unique environment (no close-up or portrait framing so
#                      OVIE lateral synthesis stays coherent).
# background_prompt  – strongly differentiated colour palette / biome so each
#                      scene is spatially distinct from every other.
WORDS_AND_PROMPTS = [
    # scenes ──────────────────────────────────────────────────────────────────
    ("CASTLE",
     "A grand medieval stone castle rising from mist-shrouded sea cliffs at golden hour, tall "
     "turrets with glowing amber windows, a drawbridge over a dark moat, ravens circling "
     "overhead, golden light on grey stone, high fantasy, photorealistic",
     "Pale grey fog-shrouded sea cliffs at cold dawn, blue-grey ocean mist rolling over dark "
     "rocks and breaking waves far below, no buildings, high fantasy, photorealistic"),

    ("FOREST",
     "An ancient enchanted forest with enormous silver-barked trees, bioluminescent mushrooms "
     "and glowing flowers on the mossy floor, magical golden light rays filtering through the "
     "dense canopy, ethereal mist between the trunks, high fantasy, photorealistic",
     "Deep moonlit silver-barked forest at night, soft blue moonlight through intertwined "
     "canopy, glowing blue-white fungi on the dark mossy ground, no golden light, high "
     "fantasy, photorealistic"),

    ("DUNGEON",
     "A dark underground dungeon with rough stone walls and iron-barred cells, flickering "
     "torchlight casting long shadows on scattered bones and rusted weapons on the damp floor, "
     "heavy chains on the walls, gothic high fantasy, photorealistic",
     "Wet black stone underground corridors with green algae on the walls, cold dripping "
     "moisture, no torches, dim greenish ambient light seeping from below, gothic high "
     "fantasy, photorealistic"),

    ("TAVERN",
     "Interior of a warm medieval fantasy tavern at night, low wooden beams, a large stone "
     "fireplace roaring with orange fire, cloaked adventurers at heavy oak tables with "
     "tankards of ale, warm amber candlelight, high fantasy, photorealistic",
     "Warm amber candlelit medieval inn corridor with low wooden beams, empty wooden benches "
     "against plastered walls hung with iron lanterns, high fantasy, photorealistic"),

    ("PORTAL",
     "A swirling circular arcane portal of violet and gold energy suspended between ancient "
     "moss-covered stone pillars in a ruin, glimpses of another starlit realm through the "
     "gateway, glowing runes carved into the stone, high fantasy, photorealistic",
     "Teal-green moss-covered ancient stone ruins in dense morning mist, crumbling arches and "
     "carved pillars barely visible through thick fog, no portal, high fantasy, photorealistic"),

    ("THRONE",
     "An imposing dark throne room with a massive obsidian throne on a raised dais, towering "
     "pillars lined with burning red braziers, tattered battle banners hanging from vaulted "
     "ceilings, dramatic crimson candlelight, high fantasy, photorealistic",
     "Deep black stone ceremonial hall with towering pillars and intense red brazier light, "
     "tattered banners and vaulted stone ceilings, no throne, high fantasy, photorealistic"),

    ("RUINS",
     "Ancient stone ruins of a fallen elven city overgrown with vines and glowing teal moss, "
     "crumbling archways and toppled statues, pale grey mist drifting under an overcast sky, "
     "high fantasy, photorealistic",
     "Pale grey ancient stone ruins under heavy overcast sky, crumbling columns and broken "
     "archways carpeted with moss, cool diffuse light, no glow, high fantasy, photorealistic"),

    ("CRYPT",
     "A vast underground crypt with rows of stone sarcophagi carved with warrior reliefs, "
     "cold blue torch sconces on damp walls, cobwebs draping the carved ceilings, scattered "
     "bones on the stone floor, high fantasy, photorealistic",
     "Bone-white stone crypt corridors lit by cold blue torchlight, carved archways and "
     "cobweb-draped columns receding into darkness, no sarcophagi, high fantasy, "
     "photorealistic"),

    ("SHRINE",
     "A moss-covered outdoor forest shrine at midday, a stone idol surrounded by offerings "
     "of candles and wildflowers, shafts of warm golden dappled light through the leafy "
     "canopy above, ancient mystical atmosphere, high fantasy, photorealistic",
     "Bright green sunlit forest glade with warm golden shafts of light through old oak "
     "canopy, carpet of ferns and wildflowers, glowing mushrooms, no shrine, high fantasy, "
     "photorealistic"),

    ("ALTAR",
     "A dark stone altar in an underground ritual chamber, carved with glowing arcane symbols, "
     "surrounded by tall burning black candles, ominous violet light from a crack in the "
     "ceiling above, high fantasy, photorealistic",
     "Dark underground ritual chamber with rough stone walls, tall black candles and violet "
     "arcane smoke drifting, carved rune inscriptions, no altar, high fantasy, photorealistic"),

    ("FORGE",
     "Interior of a dwarven forge, a massive stone furnace blazing with intense orange fire, "
     "sparks flying, glowing red-hot iron on a huge anvil, hammers and weapons hung on the "
     "stone walls, high fantasy, photorealistic",
     "Deep underground cavern with glowing orange-red lava veins running through dark basalt "
     "rock walls, heat haze shimmering in the air, no figures or forge, high fantasy, "
     "photorealistic"),

    # objects — scene level ───────────────────────────────────────────────────
    ("POTION",
     "A cluttered alchemist's workshop lined with shelves of glowing coloured potions in "
     "glass vials, a bubbling cauldron emitting rainbow smoke, dried herbs hanging from "
     "the ceiling, open ancient grimoires on the table, warm amber and violet candlelight, "
     "high fantasy, photorealistic",
     "Warm amber alchemist workshop with crowded wooden shelves of dark glass bottles, dried "
     "herb bundles and dusty tomes, brass instruments on tables, no glowing liquids, high "
     "fantasy, photorealistic"),

    ("CROWN",
     "An ancient royal crown of dark twisted gold set with glowing rubies and sapphires "
     "resting on a crimson velvet cushion atop a stone pedestal in a grand treasure vault, "
     "ornate walls with golden reliefs, dramatic side lighting from wall sconces, high "
     "fantasy, photorealistic",
     "Crimson-draped royal vault interior with carved golden wall reliefs and dim warm sconce "
     "lighting, stone archways hung with velvet drapes, no crown, high fantasy, "
     "photorealistic"),

    ("GRIMOIRE",
     "An open ancient spellbook lying on a reading lectern in a moonlit sorcerer's study, "
     "yellowed pages glowing with arcane symbols and diagrams, a quill pen resting on the "
     "page, silver moonlight through tall arched windows, high fantasy, photorealistic",
     "Silver-blue moonlit sorcerer's library with tall dark wooden shelves of ancient tomes, "
     "pale blue moonlight through leaded windows, dust motes in the air, no open book, high "
     "fantasy, photorealistic"),

    ("SWORD",
     "A legendary enchanted sword with a jewelled crossguard and glowing runes etched along "
     "the gleaming blade, thrust upright into a mossy stone plinth in a sun-dappled forest "
     "clearing, rays of golden light breaking through the oak canopy above, high fantasy, "
     "photorealistic",
     "Sun-dappled ancient forest clearing with warm golden shafts of light through old oak "
     "canopy, ferns and mossy rocks, no sword or stone, high fantasy, photorealistic"),

    # persons and creatures — scene level ─────────────────────────────────────
    ("DRAGON",
     "A colossal dragon with obsidian scales and glowing amber eyes perched on a shattered "
     "mountain peak, wings spread wide against a stormy grey sky, smoke curling from its "
     "nostrils, lightning crackling in the dark storm clouds behind it, high fantasy, "
     "photorealistic",
     "Cold grey glacier and snow-covered mountain peaks under heavy overcast sky, blue-white "
     "ice crevasses, no dragon, high fantasy, photorealistic"),

    ("WIZARD",
     "An elderly wizard in deep blue star-covered robes standing in a circular arcane "
     "laboratory, one hand raised casting a glowing golden spell, ancient books floating in "
     "the air around him, magical energy crackling from his gnarled staff, high fantasy, "
     "photorealistic",
     "Emerald green circular arcane laboratory with gleaming brass orreries and spinning "
     "magical instruments, crackling copper lightning coils, no figure, high fantasy, "
     "photorealistic"),

    ("GOBLIN",
     "A sneaky green-skinned goblin crouching on a pile of stolen treasure in a torchlit "
     "cave, large pointed ears, crooked yellow teeth gleaming, wide glinting eyes clutching "
     "a stolen jewel to its chest, scattered gold coins around it, high fantasy, "
     "photorealistic",
     "Deep red-orange torchlit cave interior with rough stone walls and glittering gems "
     "embedded in the rock, long orange torch shadows, no goblin, high fantasy, "
     "photorealistic"),

    ("KNIGHT",
     "A noble knight in gleaming silver full plate armour standing in a sunlit castle "
     "courtyard, ornate plumed helmet under one arm, determined expression, warm afternoon "
     "golden light on stone walls hung with colourful heraldic banners, high fantasy, "
     "photorealistic",
     "Sunlit stone castle courtyard with warm golden afternoon light, flagstone floor, stone "
     "walls hung with colourful heraldic banners, no figure, high fantasy, photorealistic"),

    ("ELF",
     "A wise elven warrior in intricate golden leaf armour standing in an ethereal "
     "silver-barked forest at dawn, long silver hair, piercing blue eyes, magical golden "
     "mist rising from the mossy ground around her feet, high fantasy, photorealistic",
     "Soft pale pink and gold dawn light through delicate silver-barked birch trees, wispy "
     "pink morning mist drifting between the trunks, no figure, high fantasy, photorealistic"),

    ("DWARF",
     "A stout dwarf warrior in rune-engraved bronze armour standing before a great "
     "underground forge, long braided red beard adorned with gold rings, a heavy battle axe "
     "resting on his shoulder, intense orange forge-fire glow behind him, high fantasy, "
     "photorealistic",
     "Vast cold grey underground dwarven city with colossal carved stone columns disappearing "
     "into darkness overhead, blue-white phosphorescent moss on ancient walls, no figure, "
     "high fantasy, photorealistic"),

    ("WITCH",
     "An old witch in a wide crow-feather hat seated at a candle-strewn table in her forest "
     "cottage, sharp green eyes over a steaming cauldron, dried herbs hanging from low "
     "rafters, glass jars lining every shelf, warm amber candlelight, high fantasy, "
     "photorealistic",
     "Purple-grey twilight forest glade with gnarled ancient oak trees, wisps of pale green "
     "witch-fire floating among the dark ferns, no figure, high fantasy, photorealistic"),

    ("GOLEM",
     "A massive stone golem with glowing orange eyes standing upright in a frost-covered "
     "ancient stone chamber, arcane runes etched across its cracked granite body, towering "
     "over crumbled stone blocks on a frozen floor, high fantasy, photorealistic",
     "Frost-covered ancient stone chamber with faintly glowing rune engravings on the walls "
     "and ice crystals on the floor, cold blue ambient light, no golem, high fantasy, "
     "photorealistic"),

    ("VALKYRIE",
     "A fierce valkyrie in silver winged armour standing on a golden sunset fjord cliff-top, "
     "long golden hair streaming in the wind, a glowing spear raised to the sky, dramatic "
     "warm amber light on sheer cliff faces, high fantasy, photorealistic",
     "Warm golden sunset over a dramatic Scandinavian fjord, sheer cliff faces glowing amber "
     "and gold, calm copper-coloured water far below, no figure, high fantasy, photorealistic"),

    ("SORCERER",
     "A gaunt dark sorcerer in black and deep purple robes standing at the top of an obsidian "
     "tower at twilight, glowing violet eyes, a skull-topped staff crackling with dark energy, "
     "swirling tendrils of shadow rising from his hands, high fantasy, photorealistic",
     "Deep purple twilight at the top of an obsidian stone tower, violet arcane energy "
     "crackling between dark stone spires, no figure, high fantasy, photorealistic"),

    ("TROLL",
     "A massive cave troll crouching in a bioluminescent underground cavern, jagged uneven "
     "teeth, flat wide nose, small yellow eyes, warty grey-green skin, a crude stone club "
     "resting against the rock, glowing teal crystal formations around it, high fantasy, "
     "photorealistic",
     "Dim greenish bioluminescent underground cavern with glowing teal crystal formations "
     "growing from dark stone walls, eerie green ambient light, no troll, high fantasy, "
     "photorealistic"),

    ("HYDRA",
     "A fearsome hydra with three serpent heads rearing up from the black water of a "
     "fog-shrouded swamp at night, dripping dark scales, forked tongues, glowing red eyes "
     "piercing the murk, gnarled dead trees silhouetted in the fog, high fantasy, "
     "photorealistic",
     "Dark fog-shrouded night swamp with gnarled silhouetted dead trees, still black water "
     "with faint red reflections in the mist, no creature, high fantasy, photorealistic"),

    ("PEGASUS",
     "A majestic white pegasus in full gallop above a vast pale blue dawn cloudscape, "
     "enormous feathered wings spread wide, soft golden light on its silver mane, wispy "
     "white clouds stretching to the horizon far below, high fantasy, photorealistic",
     "Pale blue pre-dawn sky with a vast white cloudscape, subtle golden glow along the "
     "cloud horizon, no creature, high fantasy, photorealistic"),

    ("PHOENIX",
     "A radiant phoenix rising from a pillar of golden fire on a dark volcanic plain at "
     "night, crimson and gold feathers blazing, fierce amber eyes, glowing embers drifting "
     "upward, distant orange lava flows in the rocky background, high fantasy, photorealistic",
     "Dark volcanic plain at night with distant rivers of orange-red lava glowing between "
     "black basalt rocks, glowing embers drifting in the dark air, no creature, high "
     "fantasy, photorealistic"),

    ("UNICORN",
     "A graceful unicorn standing in a silver moonlit meadow at the edge of an enchanted "
     "forest, pure white coat gleaming, spiraling silver horn glowing softly, magical "
     "fireflies drifting around its hooves, misty silver-barked trees beyond, high fantasy, "
     "photorealistic",
     "Silver moonlit meadow at the edge of an enchanted forest, glowing golden fireflies "
     "drifting in the moonlight, misty silver-barked trees in the distance, no creature, "
     "high fantasy, photorealistic"),
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

# Brightness fade: hold at 1.0 until d=FADE_START (fraction of half-sector),
# then cos⁴ drop to 0 at the midpoint.  Lower = more conservative (fades
# sooner); raise it if OVIE looks clean further from the reference.
# FADE_START=0.3 → full brightness within 13.5° of each reference image.
FADE_START = 0.3

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

        d = abs(delta) / (SECTOR_DEG / 2.0)
        if d <= FADE_START:
            brightness = 1.0
        else:
            d2         = (d - FADE_START) / (1.0 - FADE_START)
            brightness = np.cos(d2 * (np.pi / 2.0)) ** 4

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
        # Release PyTorch's cached GPU allocations so cuBLAS can create its
        # per-thread handle without hitting an out-of-memory error on the first
        # OVIE call (which may be the first cuBLAS op in this thread).
        torch.cuda.empty_cache()
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
