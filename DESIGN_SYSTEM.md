# Inventory Intelligence — UI/UX Design System

## Visual concept
A premium dark-mode **glassmorphism** dashboard: deep midnight surfaces, translucent cards, violet/cyan/magenta light leaks, restrained borders, and subtle depth.

## Color palette

| Token | Hex | Usage |
|---|---|---|
| `--bg` | `#080B18` | Main background |
| `--bg-2` | `#0E1224` | Background gradient endpoint |
| `--text` | `#F7F8FF` | Primary text |
| `--muted` | `#A7AFC4` | Secondary text |
| `--muted-2` | `#7C859D` | Metadata / helper text |
| `--violet` | `#9B7BFF` | Primary accent |
| `--cyan` | `#5EE7F8` | Secondary accent |
| `--magenta` | `#F36CF4` | Ambient accent |
| `--success` | `#78F5BB` | Positive state |
| `--warning` | `#FFD17A` | Reorder warning |
| `--danger` | `#FF8294` | Validation error |

### Glass surfaces
- Main glass fill: `rgba(19, 24, 45, 0.58)`
- Strong glass: `rgba(20, 25, 49, 0.78)`
- Subtle inset overlay: `rgba(255, 255, 255, 0.045)`
- Fine border: `rgba(255, 255, 255, 0.11)`
- Highlight border: `rgba(255, 255, 255, 0.18)`

## CSS tokens
The complete tokens live in `static/style.css` under `:root` and cover:
- glass backgrounds
- border opacity
- shadow layers
- corner radii
- accent colors
- typography colors

## Glass recipe
```css
background: linear-gradient(145deg, rgba(255,255,255,.055), rgba(255,255,255,.02));
background-color: rgba(19, 24, 45, .58);
border: 1px solid rgba(255,255,255,.11);
backdrop-filter: blur(24px) saturate(138%);
box-shadow: 0 10px 30px rgba(0,0,0,.23), inset 0 1px 0 rgba(255,255,255,.06);
```

## Micro-interactions
- **Ambient lighting:** three large blurred orbs slowly drift behind the content.
- **3D tilt:** cards calculate pointer position and rotate up to ~7 degrees around both axes.
- **Hover gloss:** a radial highlight follows the pointer inside glass cards.
- **CTA sweep:** the prediction button has a moving diagonal shine on hover.
- **Focus lift:** form controls raise by 1px and receive a luminous focus ring.
- **Loading feedback:** the Predict button swaps to a spinner during form submission.
- **Reduced motion:** `prefers-reduced-motion` disables nonessential animation.

## Typography
Primary stack:
`Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`

Headlines use a tight tracking value and heavy weight to create a modern product-studio feel, while labels stay compact and uppercase for dashboard hierarchy.
## Form control consistency

Native select controls use `color-scheme: dark` plus dark `option` backgrounds so the opened option list stays consistent with the glassmorphism dashboard instead of flashing a default white menu. The visible select keeps a custom dark chevron.

## Data-mining result cards
The dashboard also includes separate glass panels for the Decision Tree demand-level classification and K-Means inventory-profile assignment, plus evaluation panels for classifier metrics and the held-out clustering silhouette score. These panels display measured outputs from the saved model artifacts rather than hard-coded demo values.
