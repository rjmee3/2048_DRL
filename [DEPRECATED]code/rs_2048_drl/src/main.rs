use std::io;
use crate::game_state::*;

mod game_state;

fn main() {
    let mut board = initialize_board();
    while !is_game_over(&board) {
        print_board(&board);
        let mut action = String::new();
        println!("Enter move: ");
        io::stdin().read_line(&mut action).expect("Failed Input Read.");
        action = action.trim().to_string();

        match action.as_str() {
            "w" => move_board(&mut board, Action::MoveUp),
            "a" => move_board(&mut board, Action::MoveLeft),
            "s" => move_board(&mut board, Action::MoveDown),
            "d" => move_board(&mut board, Action::MoveRight),
            "quit" => break,
            _ => println!("ERR: Invalid Action."),
        }
    }
}