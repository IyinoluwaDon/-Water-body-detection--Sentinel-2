# Optional: get a true accuracy number

**You can skip this and still have a finished project.** Do it only if you want a number that says how often the
water map matches reality, not just how often it matches the JRC water map.

It takes 45 to 60 minutes of looking at satellite images and typing `water`, `land` or `unsure`.

1. Run the notebook to the end. It writes `outputs/reference_frame/` with about 220 points. The file hides what the
   model predicted, so your answers are unbiased.
2. In the notebook's last section, run the first optional cell. It saves a frozen copy of the point list and downloads
   `reference_frame.csv` and `reference_frame.kml`.
3. Open the `.kml` in Google Earth (earth.google.com/web: Projects, Open, Import KML file).
4. For each point `P001`, `P002`, ... look at the ground at that exact spot and type one word in the `label` column of
   the CSV: `water`, `land` or `unsure`. Use `unsure` only when you truly cannot tell (more than 10% unsure makes the
   result look better than it is). Do not change the `id`, `lon` or `lat` columns. Do not look at the model map first.
5. In the notebook, run the second optional cell and upload your finished CSV.
6. Run Step 6 and Step 7 again. A new section, "Independent reference validation", appears with your accuracy figures.
