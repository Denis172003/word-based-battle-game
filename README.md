# ⚡ Words of Power — Hackathon Challenge

Welcome to **Words of Power**, a strategic text-based battle game where **every word is a weapon** and **every move has a price**.

This project was developed as part of a **hackathon challenge** focused on strategy, decision-making, and efficient resource management.

---

## 📜 Project Overview

**Words of Power** is a turn-based game where the system and the player face off over **10 rounds**:

- Each round, the **system** generates a random challenge word.
- The **player** selects a counter-word from a predefined list to **“beat”** the system’s word.
- Every word has a **cost**; stronger words are more expensive.
- The goal: **win as many rounds as possible** while **spending the least money**.

The player is rewarded for **winning rounds**, **spending wisely**, and **outsmarting** the system!

---

## 🕹️ How to Play

1. **GET Request** — Fetch the system’s challenge word each round:  
   ```bash
   GET /get-word
