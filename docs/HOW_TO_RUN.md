# How to run this (no coding needed)

You need: a Google account, and your Earth Engine **project ID**.

## One-time: find your project ID
1. Go to <https://console.cloud.google.com> and sign in.
2. Click the project name at the top of the page. A list opens.
3. Copy the value in the **ID** column of your Earth Engine project. It looks like `my-project-123456`.
   (It is not the display name.)
4. If you have never used Earth Engine, register a project first at <https://code.earthengine.google.com/register>.
   Choose the access option appropriate to your use and review the current eligibility requirements.

## Run it
1. Go to <https://colab.research.google.com> and sign in with the same Google account.
2. Click **File → Upload notebook** and choose `notebooks/waterdet_colab.ipynb` from this project.
3. Type your project ID in the box in the second cell.
4. Click **Runtime → Run all**.
5. Approve the two Google pop-ups (Earth Engine sign-in, then Google Drive access).
6. Wait about 20 to 30 minutes. Scroll down to see the maps and numbers. `waterdet_results.zip` downloads at the end.

If any cell shows a red error, save the error message and check the relevant input file, configuration and export task status.

## What was decided for you
| Choice | Decision | Why |
|---|---|---|
| Study area | Lagos lagoon area, `[3.30, 6.40, 3.55, 6.60]` | The box in your proposal. Contains lagoon, creeks, ocean, city shadows and ponds. |
| Seasons | Dry: Dec 2025 to Feb 2026. Wet: Jun to Sep 2026 | Dry adapts the proposal's example window. Wet shows seasonal change. |
| Model labels | JRC Global Surface Water | The method in your proposal. |
| Accuracy check | Automatic against JRC, plus simple-threshold comparisons | Needs nothing from you. A true accuracy figure is optional: see `OPTIONAL_ACCURACY_CHECK.md`. |

## Why Colab and not "run this one command"
The provided workflow uses interactive Google authentication. Colab offers a browser-based environment; local Python also works, as described in the README. Account access and export quotas depend on your Earth Engine registration.
