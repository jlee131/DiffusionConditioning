---
name: ui-style-preferences
description: The user's preferred visual style for web UIs (survey pages, forms, small web apps, interactive explainers) — English headings, a clear typographic hierarchy, outline-only navigation buttons, and a sky/peach/coral palette with a dark-mode variant the user prefers. Use when building, restyling, or tweaking any web page or front-end for this user, including bilingual (English/Korean) pages.
---

# UI style preferences

Preferences collected while building a bilingual image-rating survey (2026-11) and an interactive diffusion-model explainer (2026-11). Apply them by default; the user's explicit request in the current task wins.

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
| deepened sky | `#3E9ED3` | selected rating cell (≥3:1 on white) |
| light sky | `#E3F2FA` | hovered cell, soft highlights |
| peach | `#F8BC9B` | outline of Start/Next/Submit |
| light peach | `#FDE8DC` | position badges, outline-button hover; badge text `#9A4A2A` |
| coral | `#F89B9B` | reference ("Input") card header/border, progress fill, warning border; header text `#6B2626` |
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
- Canvas code that parses colors needs hex values in the tokens, not `rgb()` or named colors.

## 4b. Dark mode (preferred: "darker looks cleaner")

Offer dark mode for every page. Deep navy surfaces; the same three hues are lifted for contrast rather than replaced.

| Token (role) | Light | Dark |
|---|---|---|
| page / surface | `#F8F9FB` / `#FFFFFF` | `#0E141B` / `#151E28` |
| body text / headings (`--strong`) | `#3A4659` / `#23384F` | `#DCE3EB` / `#F1F5F9` |
| muted text / mobile help text | `#677184` / `#8A93A3` | `#93A0B0` / `#7F8C9C` |
| borders / control borders | `#E8EAEE` / `#D9DEE5` | `#233140` / `#2F4052` |
| keywords, focus ring (deep sky) | `#2A7DAA` | `#8CCDEE` |
| selected segment fill / its text | `#3E9ED3` / `#23384F` | `#75BDE0` / `#0E141B` |
| hover tint (light sky) | `#E3F2FA` | `#1A3042` |
| badge bg / badge text (peach) | `#FDE8DC` / `#9A4A2A` | `#3A2A22` / `#F8BC9B` |
| reference header bg / text (coral) | `#F89B9B` / `#6B2626` | `#3B2326` / `#FFC2C2`, coral `#F89B9B` border |
| card shadow | `0 1px 2px rgba(35,56,79,.04)` | none |

- In dark mode, solid pastel fills become a **dark tint of the hue plus a light text in that hue** (e.g. the coral reference header, peach badges). Only the selected segment keeps a bright fill (sky with dark text).
- Mechanism: the light tokens live on `:root`; the dark tokens go under `@media (prefers-color-scheme: dark)` guarded by `:root:not([data-theme="light"])`, and again under `:root[data-theme="dark"]`. A small pill toggle (`☾ Dark` / `☀ Light`) at the top right sets `data-theme`, remembers the choice in `localStorage` (wrapped in try/catch), and accepts a `?theme=dark|light` URL override. Canvas drawings must redraw when the theme changes (listen for a custom `themechange` event as well as the media query).
- Never hard-code `#fff` or other light colors in component rules; use the role tokens so both modes follow.

## 5. Layout

- Comparison grids: four cards per row on desktop; on mobile, three per row so the reference plus two candidates fit one screen width. Exception: with exactly four large image cards (reference + three), use 2 × 2 on mobile so no card is left alone on a row.
- The reference item comes first and looks different (coral header); candidates get neutral cards with peach letter badges.
- Keep desktop and mobile rules separate (`@media (max-width: 760px)`), and when changing one, say which one changed and whether the other is affected.

## 6. Working style for UI tweaks

- Apply exactly the requested change, then report it as a short before → after table with pixel values, scoped to desktop or mobile.
- If the user says not to run tests, don't; otherwise check visually. Headless Chrome cannot render narrower than ~500px, so capture mobile widths through an iframe preview of the page at the target width.
- When trying a new look on an existing page, keep the current design intact and add the new one as an override stylesheet behind a URL switch (e.g. `?style=sky`), so the user can compare both before choosing.
- Point out side effects the user would notice (e.g. a label now truncates, mobile text ends up larger than desktop) and offer a one-line fix rather than applying it unasked.
- Remind the user when a running server must be restarted for config changes to appear.
