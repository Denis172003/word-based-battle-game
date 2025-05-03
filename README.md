# Words of Power — Strategic Word Game Engine

## 📜 Game Overview

**Decision Logic** — Select the best counter-word from the predefined list based on strategy and cost.

**POST Request** — Send your chosen word’s ID back to the server:

```bash
POST /submit-word
Content-Type: application/json

{
  "player_id": "<your_player_id>",
  "word_id":   <chosen_word_id>,
  "round_id":  <current_round>
}
```

Repeat for **10 rounds**.

Check your final standings and discounts:

```bash
GET /status?player_id=<your_player_id>
```

---

## 🧮 Scoring & Discounts

- ✅ **Win:** Pay only the word’s cost.
- ❌ **Loss:** Pay the word’s cost **plus a $75 penalty**.
- 🏆 **5% Discount** off your total for each round you win.
- 💸 **20% Cheaper-Win Refund:** If you outspend an opponent on the same round yet both win, you get **20% of your word’s cost back**.

```sql
Final Cost = ((Total Spent + 75 × Rounds Lost) × (1 – 0.05 × Rounds Won)) – Sum(Cheaper-Win Refunds)
```

---

## ⚙️ Project Structure

| File              | Purpose                                                   |
|-------------------|-----------------------------------------------------------|
| `simple_counter.py` | Orchestrates game flow: GET → decide → POST → score     |
| `logic.py`          | Decision-making engine: chooses optimal counter word    |
| `api.py`            | HTTP helper functions for GET/POST                      |
| `words.json`        | Predefined list of 77 player words & their costs        |
| `requirements.txt`  | Python dependencies                                     |
| `README.md`         | This file                                               |

---

## 🛠️ Technologies Used

- Python 3.10+
- `requests` for API interactions
- JSON for word-list configuration

---

## 🧠 Key Strategies Implemented

- **Smart Word Matching:** Select the cheapest effective counter-word.
- **Cost Optimization:** Favor low-cost words to maximize win discounts.
- **Penalty Minimization:** Spend more when necessary to avoid the $75 loss penalty.
- **Win Discounts:** Earn 5% off per round won.
- **Cheaper-Win Bonus:** 20% refund if you win with a more expensive word.

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/your-username/words-of-power.git
cd words-of-power
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Play the game:

```bash
python simple_counter.py
```

---

## ✨ Future Improvements

- 🎯 **Predictive AI:** Analyze past rounds to anticipate system words.
- 🖥️ **GUI Front-End:** Build a visual interface for more engaging gameplay.
- 🤖 **Bot Battles:** Simulate “AI vs. AI” tournaments with leaderboards.

---

## 🙌 Acknowledgments

Special thanks to the hackathon organizers for a fun and challenging event, and to all teammates who contributed!

---

## 📬 Contact

- **Email:** yourname@example.com  
- **GitHub:** [your-username](https://github.com/your-username)
