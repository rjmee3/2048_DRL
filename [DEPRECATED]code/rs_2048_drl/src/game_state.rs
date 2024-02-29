use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;
use std::process::Command;

const BOARD_SIZE: usize = 4;

pub enum Action {
    MoveLeft,
    MoveRight,
    MoveUp,
    MoveDown,
}

// initializes the board and places 2 random tiles
pub fn initialize_board()  -> [[u32; BOARD_SIZE]; BOARD_SIZE] {
    let mut board: [[u32; BOARD_SIZE]; BOARD_SIZE] = [[0; BOARD_SIZE]; BOARD_SIZE];

    place_rand_tile(&mut board);
    place_rand_tile(&mut board);

    return board;
}

// places a random tile in an empty space. 90% chance for a 2, 10% chance for a 4
fn place_rand_tile(board: &mut [[u32; BOARD_SIZE]; BOARD_SIZE]) {
    let mut zero_tiles: Vec<(usize, usize)> = Vec::new();
    let mut rng = rand::thread_rng();
    
    // acquires coords for empty tiles
    for (i, row) in board.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            if value == 0 {
                zero_tiles.push((i, j));
            }
        }
    }

    // places a 2 or 4 on an empty tile
    if let Some((row_index, col_index)) = zero_tiles.choose(&mut rng).cloned() {
        if rng.gen_range(1..=10) > 1 {
            board[row_index][col_index] = 2;
        } else {
            board[row_index][col_index] = 4;
        }
    } else {
        println!("No Zero Element Found.");
    }
}

// transposes board
fn transpose(board: &mut [[u32; BOARD_SIZE]; BOARD_SIZE]) {
    for i in 0..BOARD_SIZE {
        for j in (i+1)..BOARD_SIZE {
            let temp = board[i][j];
            board[i][j] = board[j][i];
            board[j][i] = temp;
        }
    }
}

// reflects board
fn reflect(board: &mut [[u32; BOARD_SIZE]; BOARD_SIZE]) {
    let mut temp: [[u32; BOARD_SIZE]; BOARD_SIZE] = [[0; BOARD_SIZE]; BOARD_SIZE];

    for i in 0..BOARD_SIZE {
        for j in 0..BOARD_SIZE {
            temp[i][BOARD_SIZE-1-j] = board[i][j];
        }
    }

    for i in 0..BOARD_SIZE {
        for j in 0..BOARD_SIZE {
            board[i][j] = temp[i][j];
        }
    }
}

// merges a single row
fn merge_row(row: &mut [u32; BOARD_SIZE]) {
    let mut queue: VecDeque<u32> = VecDeque::new();

    // enqueuing all non-zero elements in the row
    for &value in row.iter() {
        if value != 0 {
            queue.push_back(value);
        }
    }

    let mut index = 0;

    // dequeue elements into row, merging like elements
    while let Some(value) = queue.pop_front() {
        row[index] = value;

        // merging
        if let Some(front) = queue.front() {
            if row[index] == *front {
                row[index] += queue.pop_front().unwrap();
            }
        }

        index += 1;
    }

    // set further elements to zero
    while index < BOARD_SIZE {
        row[index] = 0;
        index += 1;
    }
}

// handles moves in all 4 directions
pub fn move_board(board: &mut [[u32; BOARD_SIZE]; BOARD_SIZE], action: Action) {
    let original_board: [[u32; BOARD_SIZE]; BOARD_SIZE] = board.clone();

    // match the action passed and move board as needed
    match action {
        Action::MoveLeft => {  // LEFT
            for i in 0..BOARD_SIZE {
                merge_row(&mut board[i]);
            }
        }
        Action::MoveRight => {  // RIGHT
            reflect(board);
            for i in 0..BOARD_SIZE {
                merge_row(&mut board[i]);
            }
            reflect(board);
        }
        Action::MoveUp => {  // UP
            transpose(board);
            for i in 0..BOARD_SIZE {
                merge_row(&mut board[i]);
            }
            transpose(board);
        }
        Action::MoveDown => {  // DOWN
            transpose(board);
            reflect(board);
            for i in 0..BOARD_SIZE {
                merge_row(&mut board[i]);
            }
            reflect(board);
            transpose(board);
        }
    }

    let mut count: usize = 0;
    for i in 0..BOARD_SIZE {
        for j in 0..BOARD_SIZE {
            if board[i][j] == original_board[i][j] {
                count += 1;
            }
        }
    }

    if count < BOARD_SIZE.pow(2) {
        place_rand_tile(board);
    }
}

// checks if game is over
pub fn is_game_over(board: &[[u32; BOARD_SIZE]; BOARD_SIZE]) -> bool{
    
    for i in 0..BOARD_SIZE {
        for j in 0..BOARD_SIZE {
            if board[i][j] == 0 {
                return false;
            }
        }
    }

    for row in board.iter() {
        if (0..BOARD_SIZE-1).any(|i| row[i] == row[i+1]) {
            return false;
        }
    }

    for col in 0..BOARD_SIZE {
        let col_val: Vec<u32> = board.iter().map(|row| row[col]).collect();
        if (0..BOARD_SIZE-1).any(|i| col_val[i] == col_val[i+1]) {
            return false;
        }
    }

    true
}

pub fn print_board(board: &[[u32; BOARD_SIZE]; BOARD_SIZE]) {
    // clearing console
    if cfg!(unix) {
        Command::new("clear").status().expect("Failed To Clear Console.");
    } else {
        Command::new("cmd").arg("/c").arg("cls").status().expect("Failed To Clear Console.");
    }

    for row in board {
        let format_row: Vec<String> = row.iter().map(|value| format!("{:5}", value)).collect();
        println!("{}", format_row.join(" "));
    }
}