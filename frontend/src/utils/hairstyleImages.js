/**
 * Hairstyle Image Mapping Utility
 * Maps hairstyle names from the ML model to their corresponding image paths
 * in /hairstyles/ directory.
 */

// Map of normalized hairstyle names to their image filenames
const hairstyleImageMap = {
  // Afro styles
  afro: "/hairstyles/Afro.jpeg",
  frohawk: "/hairstyles/frohawk.webp",
  tapered_fro: "/hairstyles/tapered_fro.webp",
  high_puff: "/hairstyles/high_puff.webp",
  twist_out: "/hairstyles/twist_out.webp",

  // Coily styles
  coily_crew_cut: "/hairstyles/coily-crew-cut.jpg",
  coily_man_bun: "/hairstyles/coily_man_bun.jpg",
  long_coils_with_headband: "/hairstyles/long_coils_with_headband.jpg",
  long_coily_layers: "/hairstyles/long_coily_layers.jpg",
  short_coils_with_fade: "/hairstyles/short_coils_with_fade.jpg",

  // Bob styles
  blunt_bob: "/hairstyles/blunt-bob.jpg",
  "chin-length_bob": "/hairstyles/chin-length_bob.jpg",
  asymmetrical_bob: "/hairstyles/blunt-bob.jpg", // fallback to blunt bob

  // Waves and curls
  beach_waves: "/hairstyles/beach-waves.webp",
  bouncy_curls: "/hairstyles/Bouncy-Curls.jpg",
  medium_waves: "/hairstyles/medium_waves.jpg",
  shoulder_waves: "/hairstyles/shoulder_waves.webp",
  "shoulder-length_waves": "/hairstyles/shoulder-length_waves.jpg",
  long_soft_waves: "/hairstyles/long_soft_waves.jpg",
  lob_with_waves: "/hairstyles/lob_with_waves.jpg",
  wavy_lob: "/hairstyles/wavy_lob.jpg",
  "wavy_mid-length": "/hairstyles/wavy_mid-length.jpg",
  "wavy_shoulder-length": "/hairstyles/wavy_shoulder-length.jpg",

  // Bangs styles
  blunt_bangs: "/hairstyles/blunt-bangs.jpg",
  curtain_bangs: "/hairstyles/curtain_bangs.jpeg",
  curtain_bangs_layers: "/hairstyles/curtain_bangs_layers.jpg",
  "side-swept_bangs": "/hairstyles/side-swept_bangs.jpg",

  // Short men's cuts
  caesar_cut: "/hairstyles/caesar-cut.webp",
  crew_cut: "/hairstyles/crew_cut.webp",
  classic_taper: "/hairstyles/classic_taper.jpg",
  french_crop: "/hairstyles/french_crop.webp",
  textured_crop: "/hairstyles/textured_crop.webp",
  messy_fringe: "/hairstyles/messy_fringe.webp",
  spiky_hair: "/hairstyles/spiky_hair.jpg",
  undercut: "/hairstyles/undercut.jpg",

  // Pompadour and slick styles
  high_pompadour: "/hairstyles/high_pompadour.jpg",
  slick_back_tapered: "/hairstyles/slick_back_tapered.webp",
  slick_back_volume: "/hairstyles/slick_back_volume.jpg",
  high_fade_with_volume: "/hairstyles/high_fade_with_volume.jpg",

  // Quiff styles
  textured_quiff: "/hairstyles/textured_quiff.webp",
  textured_quiff_medium: "/hairstyles/textured_quiff_medium.webp",

  // Side part styles
  side_part: "/hairstyles/side_part.webp",
  deep_side_part: "/hairstyles/deep_side_part.jpg",
  side_swept_fringe: "/hairstyles/side_swept_fringe.webp",
  curtains_style: "/hairstyles/curtains_style.webp",

  // Layers
  layered_cut: "/hairstyles/layered_cut.jpg",
  long_layers: "/hairstyles/long_layers.jpg",
  feathered_layers: "/hairstyles/feathered_layers.jpg",

  // Pixie
  pixie_cut: "/hairstyles/pixie_cut.webp",
  pixie_with_volume: "/hairstyles/pixie_with_volume.jpg",

  // Ponytails and updos
  high_ponytail: "/hairstyles/high_ponytail.jpg",
  sleek_ponytail: "/hairstyles/sleek_ponytail.webp",

  // Medium length flow
  "shoulder-length_flow": "/hairstyles/shoulder-length_flow.webp",
};

/**
 * Get the image URL for a hairstyle by name
 * @param {string} hairstyleName - The hairstyle name from recommendations
 * @returns {string|null} - The image path or null if not found
 */
export function getHairstyleImageUrl(hairstyleName) {
  if (!hairstyleName) return null;

  // Normalize: lowercase, replace spaces with underscores
  const normalized = hairstyleName.toLowerCase().replace(/\s+/g, "_");

  // Direct lookup
  if (hairstyleImageMap[normalized]) {
    return hairstyleImageMap[normalized];
  }

  // Try with hyphens replaced by underscores
  const withUnderscores = normalized.replace(/-/g, "_");
  if (hairstyleImageMap[withUnderscores]) {
    return hairstyleImageMap[withUnderscores];
  }

  // Try with underscores replaced by hyphens
  const withHyphens = normalized.replace(/_/g, "-");
  for (const key of Object.keys(hairstyleImageMap)) {
    if (key.replace(/_/g, "-") === withHyphens) {
      return hairstyleImageMap[key];
    }
  }

  return null;
}

/**
 * Get all available hairstyle images
 * @returns {Object} - Map of hairstyle names to image paths
 */
export function getAllHairstyleImages() {
  return { ...hairstyleImageMap };
}

export default hairstyleImageMap;
