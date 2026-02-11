# QUALITYREPORT.MD

Audited images: 5
Flagged as likely gray/low quality: 3 (60.0%)
LLM audited: 5
LLM flagged gray/low: 3 (60.0%)

## Sample flagged images
- tifa_dataset\render_3d\689d0fb2c0d4_tifa-lockhart-full-4092735.jpg | luma_std=39.21 sat_std=58.80 sat_mean=199.59 color=79.39 flat=0.02 lap=582.50 edge_den=0.1477 entropy=6.39 edge=15.94 bright=51.63 flags=llm_low | llm=low (The image is low resolution, blurry, and pixelated.)
- tifa_dataset\render_3d\40b7c1f11f84_genzo1993-016-jpeg.jpg | luma_std=36.82 sat_std=43.63 sat_mean=64.80 color=34.02 flat=0.31 lap=93.20 edge_den=0.0280 entropy=6.11 edge=6.07 bright=87.59 flags=llm_gray | llm=low (The image is corrupted or incomplete, with a large solid gray block covering the bottom portion.)
- tifa_dataset\render_3d\124581e7761c_hd-wallpaper-tifa-lockhart-final.jpg | luma_std=64.96 sat_std=71.06 sat_mean=76.28 color=45.86 flat=0.02 lap=512.49 edge_den=0.1445 entropy=7.85 edge=14.79 bright=105.64 flags=llm_low | llm=low (The image is extremely low resolution, blurry, and heavily pixelated.)

## Sample LLM-flagged images
- tifa_dataset\render_3d\689d0fb2c0d4_tifa-lockhart-full-4092735.jpg | llm=low (The image is low resolution, blurry, and pixelated.)
- tifa_dataset\render_3d\40b7c1f11f84_genzo1993-016-jpeg.jpg | llm=low (The image is corrupted or incomplete, with a large solid gray block covering the bottom portion.)
- tifa_dataset\render_3d\124581e7761c_hd-wallpaper-tifa-lockhart-final.jpg | llm=low (The image is extremely low resolution, blurry, and heavily pixelated.)