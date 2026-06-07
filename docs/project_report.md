# NeRF-LLM Scene Inspector Report

## Overview

This report summarizes a NeRF-LLM Scene Inspector run. The system is built on
Nerfstudio and LERF and is intended as a reproducible research engineering demo.

## Scene

- Scene name: `desk_scene`
- Backend: `lerf`

## Query Results

| Query | Target | Top-k Hit | Best IoU | Confidence | Warnings |
| --- | --- | --- | --- | --- | --- |
| mug | demo query | qualitative | 0.000 | 0.8862745098039215 |  |
| laptop | demo query | qualitative | 0.000 | 0.8862745098039215 |  |
| coffee | demo query | qualitative | 0.000 | 0.8862745098039215 |  |
| metallic tools | demo query | qualitative | 0.000 | 0.8862745098039215 |  |
| container | demo query | qualitative | 0.000 | 0.8862745098039215 |  |

## Evaluation Summary

| Metric | Value |
| --- | --- |
| num_queries | 5 |
| demo_video | results\demo_assets\demo_montage.gif |

## Notes

- Demo assets may be dry-run synthetic artifacts unless generated from a trained LERF config.
- This report is portfolio-ready but does not claim state-of-the-art performance.
