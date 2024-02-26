use std::io;
use crate::game::*;

mod game;

fn main() {
    let mut board = initialize_board();
    while !is_game_over(&board) {
        print_board(&board);
        let mut action = String::new();
        println!("Enter move: ");
        io::stdin().read_line(&mut action).expect("Failed Input Read.");
        action = action.trim().to_string();

        match action.as_str() {
            "w" => move_board(&mut board, MOVE_UP),
            "a" => move_board(&mut board, MOVE_LEFT),
            "s" => move_board(&mut board, MOVE_DOWN),
            "d" => move_board(&mut board, MOVE_RIGHT),
            _ => println!("ERR: Invalid Action.")
        }
    }
}