---
name: ui-style-preferences
description: The user's preferred visual style for web UIs (survey pages, forms, small web apps, interactive explainers) — English headings, a clear typographic hierarchy, outline-only navigation buttons, a sky/peach/coral palette with light mode by default plus a dark variant, and a calm, airy explainer layout. Use when building, restyling, or tweaking any web page or front-end for this user, including bilingual (English/Korean) pages.
---

# UI style preferences

Preferences collected while building a bilingual image-rating survey (2026-11) and an interactive diffusion-model explainer (2026-11). Apply them by default; the user's explicit request in the current task wins.

The diffusion explainer ("Steering Diffusion", final version 2026-11-19) is the user's favorite result so far ("역대급 마음에 든다"). For a new explainer or interactive page, start from §7, which describes that site.

## 1. Language

- Keep **big titles and short structural labels in English**, even on a translated page: page title (`User Study`), reference labels (`Input`), progress headings (`Set 1 / 4`). Translate body text, questions, explanations, buttons, and status messages.
- For Korean text: add Hangul system fonts after the Latin font, wrap at spaces, and allow synthetic bold (many Latin fonts have no Hangul glyphs, and `font-synthesis: none` then drops bold):

  ```css
  :root:lang(ko) {
    font-family: Inter, "Apple SD Gothic Neo", "Noto Sans KR", "Noto Sans CJK KR", "Malgun Gothic", sans-serif;
    word-break: keep-all;
    overflow-wrap: anywhere;
    font-synthesis: weight;
  }
  ```

- Offer a language switch as small pill links (`English` / `한국어`, each in its own language) on the first page only, on the same row as the primary action, pushed to the right edge.

## 2. Typographic hierarchy

Make each level visibly distinct. **Lower levels get smaller, regular weight (no bold), and lighter gray.** Labels/keywords may stay bold; the sentence that explains them should not.

Reference scale that the user settled on (desktop / mobile, px):

| Level | Desktop | Mobile | Weight / color |
|---|---|---|---|
| Page title (h1) | 32 | 28 | 600 |
| Lead / instruction under the title | 17 | 18 | 400, muted |
| Item title (e.g. criterion name) | 18 | 18 | 600 |
| Item text (question) | 16 | 15 | 400 |
| Detail / help text | 14 | 13 | 400, muted (mobile: lighter, `#8A93A3`) |
| Keyword label (Parts, Structure…) | 16 | 16 | 700, accent text color |
| Explanation beside a keyword | 15 | 14 | 400 |
| Buttons | 14 | 14 | 600 |
| Card badge (A, B, C…) | 14 in a 29px box | 12 in a 22px box | 600 |

- Spacing: 8px between an item title and the text under it; ~10px vertical padding between list items on mobile.
- Change sizes in small steps (1–2px). When asked to "enlarge for readability", ~1.2× is right; 1.5× was too large.
- Give text room: a wide text column (~830px) on intro pages rather than a narrow centered block.

## 3. Buttons and controls

- **Navigation buttons that move to another page/step (Start, Next, Submit) are outline-only:** white fill, 2px colored border (peach), navy text; on hover, a light tint of the border color. No solid fills.
- Secondary actions (Back, Edit responses): neutral 1px border, white fill.
- Rating controls: a compact horizontal segmented bar; the selected cell is filled with the deepened accent. Put end labels (e.g. `Very poor` / `Excellent`) below the last bar on mobile.
- Help: a small round `?` icon that opens a bubble (hover on desktop, tap to pin). On mobile, place it **right beside the keyword**, with the explanation on the next line.
- Remove decorative elements that add no information (e.g. a sample rating bar on the intro page).

## 4. Color palette (sky, peach, coral — no sand)

One role per hue so no color dominates. Text is navy slate on a near-white page.

| Token | Value | Role |
|---|---|---|
| sky | `#75BDE0` | primary accent, active `?` icon |
| deep sky | `#2A7DAA` | keywords, question numbers, focus ring |
| deepened sky | `#3E9ED3` | slider/checkbox accent; selected rating cell on the survey (≥3:1 on white) |
| pale sky | `#A9D8F0` | selected segment-button fill on the explainer, with navy text (`#3E9ED3` behind navy text was too dark to read) |
| light sky | `#E3F2FA` | hovered cell, soft highlights |
| peach | `#F8BC9B` | outline of Start/Next/Submit |
| light peach | `#FDE8DC` | position badges, outline-button hover; badge text `#9A4A2A` |
| coral | `#F89B9B` | reference ("Input") card header/border, progress/loading fill, warning border; header text `#6B2626` |
| navy slate | `#3A4659` | body text; navy `#23384F` for text on sky (white on sky is too faint) |
| muted | `#677184` | secondary text |
| page / surface | `#F8F9FB` / `#FFFFFF` | background / cards |
| borders | `#E8EAEE`, controls `#D9DEE5` | |

Define all colors as CSS variables in `:root` and name them by role (`--accent`, `--next-border`, `--reference-bg`, …). Do not introduce sand/yellow unless asked.

### Data-visualization colors

Pastel tokens are too faint for small marks (dots, 2px chart lines), so data colors use deepened versions of the same hues:

| Role | Light | Dark |
|---|---|---|
| series 1 (e.g. CFG, "pencil" class) | deep sky `#2A7DAA` | sky `#75BDE0` |
| series 2 (e.g. CG, guidance arrows) | coral `#E0666B` | `#F48A8F` |
| series 3 (e.g. TFG) | **lavender** `#7E6BD6` | `#B7A6F2` |
| category in peach (e.g. "photo" class) | `#E8895A` | `#F8A97E` |
| neutral / baseline series | `#8A93A3` | `#93A0B0` |

- Peach next to coral is hard to tell apart in charts, so a third data series uses **lavender**. Lavender is the one approved hue outside the sky/peach/coral set; it is still not sand or yellow.
- Keep a method's color the same everywhere it appears (buttons, chart lines, legends, card accents).
- Density / heatmap backgrounds: a soft sky tint (`#7CC3E6` fading to the card color) in light mode, not gray ("gray isn't pretty"); in dark mode a pale ink tint (`#DCE3EB`) on navy.
- Contour lines: the same hue as the arrows they explain, thickening with the level (e.g. 0.25 / 0.5 / 0.75 thin → thick) instead of using different colors.
- Canvas code that parses colors needs hex values in the tokens, not `rgb()` or named colors.

## 4b. Light default, dark variant

**Light mode is the default** (decided on the explainer: "light mode turned out better than expected"). Still offer a dark mode on every page; the user likes it ("darker looks cleaner") but as an option, not the first view. Deep navy surfaces; the same three hues are lifted for contrast rather than replaced.

| Token (role) | Light | Dark |
|---|---|---|
| page / surface | `#F8F9FB` / `#FFFFFF` | `#0E141B` / `#151E28` |
| body text / headings (`--strong`) | `#3A4659` / `#23384F` | `#DCE3EB` / `#F1F5F9` |
| muted text / mobile help text | `#677184` / `#8A93A3` | `#93A0B0` / `#7F8C9C` |
| borders / control borders | `#E8EAEE` / `#D9DEE5` | `#233140` / `#2F4052` |
| keywords, focus ring (deep sky) | `#2A7DAA` | `#8CCDEE` |
| selected segment fill / its text | `#A9D8F0` / `#23384F` | `#75BDE0` / `#0E141B` |
| unselected segment text (a step lighter than body) | `#5D687B` | `#B4BFCC` |
| segment track | `#EEF1F5` | `#0E141B` |
| theme toggle bg / hover / text (Montage grays) | `#F1F1F2` / `#E1E2E4` / `#2E2F33` | `#2B2C2F` / `#37383C` / `#DBDCDF` |
| hover tint (light sky) | `#E3F2FA` | `#1A3042` |
| badge bg / badge text (peach) | `#FDE8DC` / `#9A4A2A` | `#3A2A22` / `#F8BC9B` |
| reference header bg / text (coral) | `#F89B9B` / `#6B2626` | `#3B2326` / `#FFC2C2`, coral `#F89B9B` border |
| card shadow | `0 1px 2px rgba(35,56,79,.04)` | none |

- In dark mode, solid pastel fills become a **dark tint of the hue plus a light text in that hue** (e.g. the coral reference header, peach badges). Only the selected segment keeps a bright fill (sky with dark text).
- Mechanism: the light tokens live on `:root`; the dark tokens go under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. A head script sets `data-theme` before first paint: `?theme=dark|light`, else the choice saved in `localStorage` (wrapped in try/catch), else `light`; `?theme=system` leaves it unset so the OS setting applies. A small borderless gray chip toggle (`☾ Dark` / `☀ Light`, 10px radius, Pretendard 13/600, Montage-style) at the top right flips and saves it. Canvas drawings must redraw when the theme changes (listen for a custom `themechange` event as well as the media query) and after web fonts load.
- Theme-dependent cursors: the system crosshair can turn white on a light canvas (Windows draws it inverted). Use a custom SVG crosshair token: a dark line with a light halo in light mode, the reverse in dark mode.
- Never hard-code `#fff` or other light colors in component rules; use the role tokens so both modes follow.

## 5. Layout

- Comparison grids: four cards per row on desktop; on mobile, three per row so the reference plus two candidates fit one screen width. Exception: with exactly four large image cards (reference + three), use 2 × 2 on mobile so no card is left alone on a row.
- The reference item comes first and looks different (coral header); candidates get neutral cards with peach letter badges.
- Keep desktop and mobile rules separate (`@media (max-width: 760px)`), and when changing one, say which one changed and whether the other is affected.
- **No horizontal page scroll on phones.** Single-column grids use `grid-template-columns: minmax(0, 1fr)` (a plain `1fr` grows to the widest equation or canvas), cards get `min-width: 0`, and wide equations scroll inside their own card. Measure at 375–390px width before reporting.

## 6. Working style for UI tweaks

- Apply exactly the requested change, then report it as a short before → after table with pixel values, scoped to desktop or mobile.
- If the user says not to run tests, don't; otherwise check visually. Headless Chrome cannot render narrower than ~500px, so capture mobile widths through an iframe preview of the page at the target width.
- When trying a new look on an existing page, keep the current design intact and add the new one as an override stylesheet behind a URL switch (e.g. `?style=sky`), so the user can compare both before choosing.
- Point out side effects the user would notice (e.g. a label now truncates, mobile text ends up larger than desktop) and offer a one-line fix rather than applying it unasked.
- Remind the user when a running server must be restarted for config changes to appear.
- After a deploy, tell the user to check in a private window: static hosts cache JS (GitHub Pages: 10 min), and modules imported by other modules can survive Ctrl+F5.

## 7. Interactive explainer pattern (the favorite site)

Reference implementation: `site/` of the diffusion-guidance project (`theme-sky.css` on top of `main.css`).

**Type and spacing**
- Fonts: two families with split roles (adopted from Wanted's Montage design system, github.com/wanteddev/montage-web, after a side-by-side test page):
  - **Pretendard** (variable, jsDelivr `orioncactus/pretendard@v1.3.9`) for headings and all UI text: h1–h3, chart titles, buttons, selects, control labels, fold titles, the loading status. One token, `--ui-font`.
  - **Source Sans 3** (400/600; Light 300 was tried and dropped) for prose: lead, body, captions, equation-card text. Canvas text follows the body font.
  - Only the *font* of UI text was taken from Montage, not its sizes; control text keeps the explainer sizes (14px labels).
- Heading scale (Montage display1 / title1; desktop / mobile, px): h1 56/72 → 40/52, weight 700, letter-spacing −0.0319em (−0.0282em on mobile), margin 88 above, 54 below; section h2 32/44 → 24/32, weight 700, letter-spacing −0.0253em (−0.023em), with a round numbered badge (pale sky circle, deep-sky number); h3 18/26 (600, −0.002em). Lead 17 / 18 muted; body 16; captions 14 / 13 muted; chart titles 17 (600).
- Airy rhythm: generous hero padding, and sections separated by a thin divider line with 88px above and below (56px on mobile).
- Text column max ~830px; cards and grids may be wider (max page 1180px).

**Structure of each section**
- Heading → one short lead paragraph → the interactive stage (canvas card, about 2/3 width) beside a control panel card (~320px) → charts and an equation card underneath.
- Each chart has a short title above it (e.g. "Guidance strength") and a muted caption below.
- Segmented controls (Montage shape, sky colors): a pale track (12px radius, 4px padding, no dividers between items), items with 8px radius, the selected item a raised pale-sky block with navy text and a hairline shadow (`0 1px 2px -1px rgba(23,23,23,.10)`); unselected labels a step lighter than body text. Used for every one-of-many choice (method, model, background, guidance scale, guidance start).
- Buttons: secondary buttons (e.g. "New noise") get the Montage shape: 10px radius, control-line border, semi-bold text. The main action button ("Generate") keeps the peach outline and is bold (700).
- Checkboxes are drawn in CSS (`appearance: none`, 16px, 4px radius, deepened-sky fill when checked) so the tick is always white; the native `accent-color` picks a dark tick on light sky.
- Control panel: segmented buttons for discrete choices, labelled sliders showing the value in monospace (`s = 1.50`), and **folds** (`▸ Display settings`, `▸ Field values`, `▸ Advanced: …`) for anything a first-time reader does not need. Keep the default view minimal.
- Legends show only what is currently drawn; turning a layer off hides its legend entry too.
- Defaults start from the plain, unguided state (method None, time 0) so the reader turns things on step by step.

**Controls and feedback**
- Icon buttons for tool actions (↶ undo, ⟳ clear) as inline SVG with `aria-label`, not text buttons.
- Long work (model download, warm-up) gets a thin progress bar with a short status line; the bar is coral so it is not mistaken for a slider.
- Final status messages are one short line ("Done in 7.2 s · scrub the slider to replay"), not paragraphs.
- Result cards are left-aligned (`auto-fill`, `justify-content: start`), not centered.
- A drawing pad: the card is exactly as wide as the pad, with a visible dashed border that darkens on hover, plus the custom crosshair above.
- An example image next to the intro ("Made on this page.") shows what the tool can produce.

**Tech choices that kept it clean**
- Plain HTML + ES modules, no build step; pinned CDN libraries (KaTeX, TF.js); all charts hand-drawn on Canvas 2D with theme tokens.
- Heavy work is chunked or deferred (idle callback + IntersectionObserver) so the page never freezes on load.
