import requests
from time import sleep
import random
from simple_counter import WordsOfPowerGame # Assuming your class is in simple_counter.py

# --- Configuration ---
HOST = "http://192.168.200.127:8000" # <<< SET YOUR SERVER ADDRESS HERE
PLAYER_ID = "H89LHUW1" # <<< SET YOUR TEAM ID HERE
NUM_ROUNDS = 10 # Target number of rounds to play
# --- End Configuration ---

POST_URL = f"{HOST}/submit-word"
GET_URL = f"{HOST}/get-word"
STATUS_URL = f"{HOST}/status"
REGISTER_URL = f"{HOST}/register"

def register(player_id):
    """Registers the player ID with the tournament server."""
    print(f"Registering player ID: {player_id}...")
    try:
        response = requests.post(REGISTER_URL, json={"player_id": player_id}, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        print("Registration successful:")
        print(response.json())
        return True
    except requests.exceptions.RequestException as e:
        print(f"Registration failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                print(f"Server response: {e.response.json()}")
            except requests.exceptions.JSONDecodeError:
                print(f"Server response (non-JSON): {e.response.text}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during registration: {e}")
        return False

def get_best_word_id(game_instance, system_word):
    """Finds the best counter word ID using the WordsOfPowerGame logic."""
    if not system_word:
        print("Warning: Received empty system word. Choosing a random low-cost word.")
        # Fallback: choose a random word from the library (e.g., lowest cost ones)
        try:
            # Example: pick a random word with cost <= 20
            low_cost_words = [word for word, data in game_instance.word_library.items() if data['cost'] <= 20]
            if not low_cost_words: # If none found, pick any random
                 low_cost_words = list(game_instance.word_library.keys())
            chosen_word_name = random.choice(low_cost_words)
            print(f"Fallback choice: {chosen_word_name}")
            return game_instance.word_library[chosen_word_name]['id']
        except Exception as e:
             print(f"Error during fallback word selection: {e}. Returning ID 1.")
             return 1 # Absolute fallback

    print(f"Finding counter for system word: '{system_word}'")
    result = game_instance._select_best_counter(system_word) # Use the internal method directly
    best_counter_info = result.get("best")

    if best_counter_info:
        chosen_word_name = best_counter_info['word']
        chosen_word_id = game_instance.word_library[chosen_word_name]['id']
        print(f"Chosen counter: {chosen_word_name} (ID: {chosen_word_id}, Cost: {best_counter_info['cost']}, Score: {best_counter_info['score']:.4f})")
        return chosen_word_id
    else:
        print("Warning: find_counter did not return a best word. Choosing random.")
        # Fallback if the logic fails unexpectedly
        random_word_name = random.choice(list(game_instance.word_library.keys()))
        print(f"Random fallback choice: {random_word_name}")
        return game_instance.word_library[random_word_name]['id']

def play_game(player_id, game_instance):
    """Runs the game loop, interacting with the server and reacting to its round number."""
    print("\n--- Starting Game ---")
    processed_rounds = set() # Keep track of rounds we've already submitted for

    while len(processed_rounds) < NUM_ROUNDS:
        current_sys_word = None
        current_round_num = -1
        game_over = False

        try:
            # --- Get current game state from server ---
            print(f"Polling for current word and round...")
            response = requests.get(GET_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            print(f"Received data: {data}")

            current_sys_word = data.get('word')
            current_round_num = data.get('round', -1)
            # Check for game over condition if the server provides it
            # game_over = data.get('game_over', False) # Uncomment if server sends game_over in GET /get-word

            # --- Check if it's a new round we haven't processed ---
            if current_round_num > 0 and current_round_num not in processed_rounds:
                print(f"\n--- Processing Round {current_round_num} ---")
                print(f"System word: '{current_sys_word}'")

                # Optional: Get status of the *previous* round if available
                if current_round_num > 1:
                    try:
                        print(f"Getting status for previous round ({current_round_num - 1})...")
                        status_response = requests.get(STATUS_URL, timeout=10)
                        status_response.raise_for_status()
                        status_data = status_response.json()
                        print("Previous Round Status:")
                        print(status_data)
                        # Check for game over condition from status if needed
                        # game_over = status_data.get('game_over', False)
                    except requests.exceptions.RequestException as e:
                        print(f"Could not get status: {e}")
                    except Exception as e:
                         print(f"An unexpected error occurred while getting status: {e}")

                # --- Choose and submit word ---
                chosen_word_id = get_best_word_id(game_instance, current_sys_word)
                submit_data = {"player_id": player_id, "word_id": chosen_word_id, "round_id": current_round_num}
                print(f"Submitting word ID {chosen_word_id} for round {current_round_num}...")
                try:
                    submit_response = requests.post(POST_URL, json=submit_data, timeout=15)
                    submit_response.raise_for_status()
                    print("Submission successful:")
                    print(submit_response.json())
                    processed_rounds.add(current_round_num) # Mark round as processed *after* successful submission
                    print(f"Completed rounds: {len(processed_rounds)}/{NUM_ROUNDS}")
                except requests.exceptions.RequestException as e:
                    print(f"Error submitting word: {e}")
                    if hasattr(e, 'response') and e.response is not None:
                         try:
                             print(f"Server response: {e.response.json()}")
                         except requests.exceptions.JSONDecodeError:
                             print(f"Server response (non-JSON): {e.response.text}")
                    # Decide if you want to retry or skip the round on submission failure
                except Exception as e:
                     print(f"An unexpected error occurred during submission: {e}")

            elif current_round_num in processed_rounds:
                pass
            elif current_round_num == 0:
                 print("Game hasn't started yet (round 0). Waiting...")
            else:
                 print(f"Received invalid round number {current_round_num}. Waiting...")

            # --- Check for game over condition ---
            # if game_over:
            #     print("Server indicated game over.")
            #     break # Exit the while loop

        except requests.exceptions.RequestException as e:
            print(f"Error polling server: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during polling: {e}")

        # --- Wait before polling again ---
        sleep_time = 1 # Adjust sleep time as needed
        print(f"Waiting {sleep_time} seconds...")
        sleep(sleep_time)

    print(f"\n--- Game Finished (Processed {len(processed_rounds)} rounds) ---")

if __name__ == "__main__":
    # 1. Initialize your game logic
    print("Initializing Words of Power logic...")
    try:
        game = WordsOfPowerGame()
        print("Game logic initialized.")
    except Exception as e:
        print(f"FATAL: Could not initialize WordsOfPowerGame: {e}")
        exit(1)

    # 2. Register your team
    if not register(PLAYER_ID):
         print("Registration failed. Continuing anyway...") # Decide if you want to exit
         # exit(1) # Uncomment to exit if registration fails

    # 3. Start the game loop
    play_game(PLAYER_ID, game)