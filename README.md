# NBA Player Projection and Hall of Fame Modeling

This project builds a data-driven system for evaluating NBA players and estimating Hall of Fame probability. It combines historical performance data, aging curve modeling, and machine learning to capture both peak value and long-term career trajectories.

The goal is not just to classify past players, but to provide a framework for evaluating active players whose careers are still unfolding.

---

## Overview

Most Hall of Fame models rely on static career totals or awards. This approach focuses on how players develop over time.

The pipeline:

- Tracks year-over-year changes in player performance  
- Builds empirical aging curves from historical data  
- Engineers features that emphasize peak production and efficiency  
- Uses an ensemble model to estimate Hall of Fame likelihood  

This allows the model to account for both longevity and peak performance in a consistent way.

---

## Data

The project uses season-level NBA data stored in a SQLite database. Each row represents a player-season and includes:

- Box score stats  
- Advanced metrics (BPM, VORP, WS/48, TS%)  
- Age and season context  
- Player size/archetype labels  

Counting stats are normalized to per-75 possessions to allow comparisons across eras.

---

## Aging Curve Model

Aging curves are built using a delta-based method.

For each player with consecutive seasons:

- Identify seasons at age N and N+1  
- Compute the change in each stat  
- Aggregate those changes across all players at each age  

This produces an empirical estimate of how performance evolves with age.

Curves are segmented by size group (Guard, Wing, Big) to reflect differences in career trajectories.

These curves are later used to project future performance for active players.

---

## Feature Engineering

Player evaluation is based on a combination of peak performance and career context.

Key features include:

- Peak windows (best 3-year and 7-year stretches)  
- Efficiency metrics (TS%, WS/48)  
- Impact metrics (BPM, VORP)  
- Per-75 possession normalization  
- Championship and team success context  

The focus is on capturing how dominant a player was at their best, not just cumulative totals.

---

## Archetypes

Players are grouped using clustering to capture differences in play style.

- K-means clustering is used to define modern archetypes  
- Separate handling is included for earlier eras with different statistical profiles  

This allows comparisons within more meaningful peer groups rather than across fundamentally different roles.

---

## Model

The classification step uses an ensemble of:

- Logistic regression  
- Random forest  
- Gradient boosting  

Predictions are combined using soft voting.

The model outputs a probability that a given player profile corresponds to a Hall of Fame career.

---

## Career Projection

For active players, the system applies aging curves to estimate future production.

Projected stats are then passed through the same feature pipeline and model.

This provides a forward-looking estimate rather than penalizing players who have not yet completed their careers.

---

## Outputs

The project produces:

- Aging curve datasets (per-age performance changes)  
- Player-level feature tables  
- Model predictions for Hall of Fame probability  
- Plots of aging curves by stat and player type  

Example outputs are saved in the `data/processed` directory.

---

## Project Structure
