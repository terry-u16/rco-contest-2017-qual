use std::fmt::Display;

use annealing::{Annealer, Neighbor, SingleScore};
use data_structures::FastClearArray;
use grid::{ConstMap2d, CoordIndex};
use proconio::marker::Chars;
#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;
use rand_distr::WeightedIndex;
use rand_pcg::Pcg64Mcg;

use crate::grid::{Coord, ADJACENTS};

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    map: ConstMap2d<u32, { Input::N }>,
    graph: ConstMap2d<[Option<CoordIndex>; 4], { Input::N }>,
}

impl Input {
    const N: usize = 50;
    const K: usize = 8;

    fn read_input() -> Self {
        input! {
            _h: usize,
            _w: usize,
            _k: usize,
        }

        let mut map = ConstMap2d::with_default();

        for row in 0..Self::N {
            input! {
                s: Chars,
            }

            for col in 0..Self::N {
                map[Coord::new(row, col)] = s[col].to_digit(10).unwrap();
            }
        }

        let mut graph: ConstMap2d<[Option<CoordIndex>; 4], { Input::N }> =
            ConstMap2d::with_default();

        for row in 0..Self::N {
            for col in 0..Self::N {
                let c = Coord::new(row, col);

                for (i, &adj) in ADJACENTS.iter().enumerate() {
                    let next = c + adj;

                    if next.in_map(Self::N) {
                        graph[c][i] = Some(CoordIndex::new(next.to_index(Self::N)));
                    }
                }
            }
        }

        Self { map, graph }
    }
}

fn main() {
    let input = Input::read_input();
    let state = State::new();
    let neigh_gen = NeighborGenerator;
    let annealer = Annealer::new(1e6, 1e3, 42, 1024, get_callbacks());

    let (state, diagnostics) = annealer.run(&input, state, &neigh_gen, 9.98);
    eprintln!("{}", diagnostics);
    eprintln!("Score = {}", state.calc_actual_score());
    println!("{}", state);
}

fn get_callbacks() -> Vec<Box<dyn Fn(&Input, &State, &annealing::AnnealingStatistics)>> {
    #[cfg(feature = "local")]
    {
        vec![Box::new(|_input, state, diagnostics| {
            if diagnostics.all_iter % 100000 == 0 {
                println!("{}", state);
            }
        })]
    }
    #[cfg(not(feature = "local"))]
    {
        vec![]
    }
}

#[derive(Debug, Clone)]
struct State {
    pieces: Vec<Option<Piece>>,
    piece_set: Vec<usize>,
    piece_map: ConstMap2d<Option<usize>, { Input::N }>,
    score: u32,
    visited: FastClearArray,
    rng: Pcg64Mcg,
}

impl State {
    const PIECE_BUFFER_SIZE: usize = 320;

    fn new() -> Self {
        let pieces = vec![None; Self::PIECE_BUFFER_SIZE];
        let piece_set = (0..Self::PIECE_BUFFER_SIZE).rev().collect();
        let piece_map = ConstMap2d::with_default();
        let score = 0;
        let visited = FastClearArray::new(Input::N * Input::N);
        let rng = Pcg64Mcg::from_entropy();

        Self {
            pieces,
            piece_set,
            piece_map,
            score,
            visited,
            rng,
        }
    }

    fn calc_actual_score(&self) -> u32 {
        (self.score + 9999) / 10000
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pieces = self.pieces.iter().flatten().collect::<Vec<_>>();
        write!(f, "{}", pieces.len())?;

        for piece in pieces {
            for &c in piece.piece.iter() {
                let c = c.to_coord(Input::N);
                write!(f, "\n{} {}", c.row() + 1, c.col() + 1)?;
            }
        }

        Ok(())
    }
}

impl annealing::State for State {
    type Env = Input;

    type Score = SingleScore;

    fn score(&self, _env: &Self::Env) -> Self::Score {
        SingleScore(self.score as i64)
    }
}

static mut QUEUE_BUF: Vec<CoordIndex> = Vec::new();
static mut WEIGHTS_BUF: Vec<u32> = Vec::new();

struct RandomBreakBfsNeigh {
    start: CoordIndex,
    piece: Piece,
    piece_index: usize,
    remove_index: Vec<usize>,
    ok: bool,
}

impl RandomBreakBfsNeigh {
    fn new(state: &State, start: CoordIndex) -> Self {
        let piece = Piece {
            piece: [CoordIndex(!0); Input::K],
            score: 0,
        };
        let remove_index = vec![];
        let piece_index = state.piece_set.last().copied().unwrap();

        Self {
            start,
            piece,
            piece_index,
            remove_index,
            ok: false,
        }
    }
}

impl Neighbor for RandomBreakBfsNeigh {
    type Env = Input;
    type State = State;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        if let Some(index) = state.piece_map[self.start] {
            self.remove_index.push(index);
        }

        state.visited.clear();
        let queue = unsafe { &mut QUEUE_BUF };
        let weights = unsafe { &mut WEIGHTS_BUF };
        queue.clear();
        weights.clear();
        queue.push(self.start);
        weights.push(env.map[self.start]);
        let mut piece = [CoordIndex(!0); Input::K];

        for i in 0..Input::K {
            let Ok(dist) = WeightedIndex::new(weights.iter()) else {
                return;
            };

            let index = dist.sample(&mut state.rng);
            let c = queue.swap_remove(index);
            _ = weights.swap_remove(index);

            piece[i] = c;
            state.visited.set_true(c.0);

            if let Some(i) = state.piece_map[c] {
                if !self.remove_index.contains(&i) {
                    self.remove_index.push(i);
                }
            }

            for &next in env.graph[c].iter().flatten() {
                if !state.visited.get(next.0) && !queue.contains(&next) {
                    queue.push(next);
                    weights.push(env.map[next]);
                }
            }
        }

        self.piece = Piece::new(piece, env);
        self.ok = true;
    }

    fn eval(
        &mut self,
        _env: &Self::Env,
        state: &Self::State,
        _progress: f64,
        _threshold: f64,
    ) -> Option<<Self::State as annealing::State>::Score> {
        if !self.ok {
            return None;
        }

        let mut score = state.score;

        for &i in self.remove_index.iter() {
            score -= state.pieces[i].unwrap().score;
        }

        score += self.piece.score;

        Some(SingleScore(score as i64))
    }

    fn postprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
        assert_eq!(state.piece_set.pop().unwrap(), self.piece_index);

        state.score += self.piece.score;

        for &i in self.remove_index.iter() {
            let piece = &state.pieces[i].unwrap();
            state.score -= piece.score;
            state.piece_set.push(i);
            state.pieces[i] = None;

            for &c in piece.piece.iter() {
                state.piece_map[c] = None;
            }
        }

        for &c in self.piece.piece.iter() {
            state.piece_map[c] = Some(self.piece_index);
        }

        state.pieces[self.piece_index] = Some(self.piece);
    }

    fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }
}

struct RandomBfsNeigh {
    start: CoordIndex,
    piece: Piece,
    piece_index: usize,
    remove_index: Option<usize>,
    ok: bool,
}

impl RandomBfsNeigh {
    fn new(state: &State, start: CoordIndex) -> Self {
        let piece = Piece {
            piece: [CoordIndex(!0); Input::K],
            score: 0,
        };
        let piece_index = state.piece_set.last().copied().unwrap();

        Self {
            start,
            piece,
            piece_index,
            remove_index: state.piece_map[start],
            ok: false,
        }
    }
}

impl Neighbor for RandomBfsNeigh {
    type Env = Input;
    type State = State;

    fn preprocess(&mut self, env: &Self::Env, state: &mut Self::State) {
        state.visited.clear();
        let queue = unsafe { &mut QUEUE_BUF };
        let weights = unsafe { &mut WEIGHTS_BUF };
        queue.clear();
        weights.clear();
        queue.push(self.start);
        weights.push(env.map[self.start]);
        let mut piece = [CoordIndex(!0); Input::K];

        for i in 0..Input::K {
            let Ok(dist) = WeightedIndex::new(weights.iter()) else {
                return;
            };

            let index = dist.sample(&mut state.rng);
            let c = queue.swap_remove(index);
            _ = weights.swap_remove(index);

            piece[i] = c;
            state.visited.set_true(c.0);

            for &next in env.graph[c].iter().flatten() {
                if !state.piece_map[next].is_some() && !state.visited.get(next.0) && !queue.contains(&next) {
                    queue.push(next);
                    weights.push(env.map[next]);
                }
            }
        }

        self.piece = Piece::new(piece, env);
        self.ok = true;
    }

    fn eval(
        &mut self,
        _env: &Self::Env,
        state: &Self::State,
        _progress: f64,
        _threshold: f64,
    ) -> Option<<Self::State as annealing::State>::Score> {
        if !self.ok {
            return None;
        }

        let mut score = state.score;

        if let Some(i) = self.remove_index {
            score -= state.pieces[i].unwrap().score;
        }

        score += self.piece.score;

        Some(SingleScore(score as i64))
    }

    fn postprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
        assert_eq!(state.piece_set.pop().unwrap(), self.piece_index);

        state.score += self.piece.score;

        if let Some(i) = self.remove_index {
            let piece = &state.pieces[i].unwrap();
            state.score -= piece.score;
            state.piece_set.push(i);
            state.pieces[i] = None;

            for &c in piece.piece.iter() {
                state.piece_map[c] = None;
            }
        }

        for &c in self.piece.piece.iter() {
            state.piece_map[c] = Some(self.piece_index);
        }

        state.pieces[self.piece_index] = Some(self.piece);
    }

    fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
        // do nothing
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Piece {
    piece: [CoordIndex; Input::K],
    score: u32,
}

impl Piece {
    fn new(piece: [CoordIndex; Input::K], input: &Input) -> Self {
        let mut score = 1;

        for &c in piece.iter() {
            score *= input.map[c];
        }

        Self { piece, score }
    }
}

struct NeighborGenerator;

impl annealing::NeighborGenerator for NeighborGenerator {
    type Env = Input;
    type State = State;

    fn generate(
        &self,
        _env: &Self::Env,
        state: &Self::State,
        rng: &mut impl Rng,
    ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>> {
        if rng.gen_bool(0.8) {
            let row = rng.gen_range(0..Input::N);
            let col = rng.gen_range(0..Input::N);
            let start = CoordIndex::new(Coord::new(row, col).to_index(Input::N));
            Box::new(RandomBreakBfsNeigh::new(state, start))
        } else {
            let row = rng.gen_range(0..Input::N);
            let col = rng.gen_range(0..Input::N);
            let start = CoordIndex::new(Coord::new(row, col).to_index(Input::N));
            Box::new(RandomBfsNeigh::new(state, start))
        }
    }
}

#[allow(dead_code)]
mod annealing {
    //! 焼きなましライブラリ

    use itertools::Itertools;
    use rand::Rng;
    use rand_pcg::Pcg64Mcg;
    use std::{
        fmt::{Debug, Display},
        time::Instant,
    };

    /// 焼きなましの状態
    pub trait State {
        type Env;
        type Score: Score + Clone + PartialEq + Debug;

        /// 生スコア（大きいほど良い）
        fn score(&self, env: &Self::Env) -> Self::Score;
    }

    pub trait Score {
        /// 焼きなまし用スコア（大きいほど良い）
        /// デフォルトでは生スコアをそのまま返す
        fn annealing_score(&self, _progress: f64) -> f64 {
            self.raw_score() as f64
        }

        /// 生スコア
        fn raw_score(&self) -> i64;
    }

    /// 単一の値からなるスコア
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct SingleScore(pub i64);

    impl Score for SingleScore {
        fn raw_score(&self) -> i64 {
            self.0
        }
    }

    /// 焼きなましの近傍
    ///
    /// * 受理パターンの流れ: `preprocess()` -> `eval()` -> `postprocess()`
    /// * 却下パターンの流れ: `preprocess()` -> `eval()` -> `rollback()`
    pub trait Neighbor {
        type Env;
        type State: State<Env = Self::Env>;

        /// `eval()` 前の変形操作を行う
        fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

        /// 変形後の状態の評価を行う
        ///
        /// # Arguments
        ///
        /// * `env` - 環境
        /// * `state` - 状態
        /// * `progress` - 焼きなましの進捗（[0, 1]の範囲をとる）
        /// * `threshold` - 近傍採用の閾値。新しいスコアがこの値を下回る場合はrejectされる
        ///
        /// # Returns
        ///
        /// 現在の状態のスコア。スコアが `threshold` を下回ることが明らかな場合は `None` を返すことで評価の打ち切りを行うことができる。
        ///
        /// 評価の打ち切りについては[焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)を参照。
        fn eval(
            &mut self,
            env: &Self::Env,
            state: &Self::State,
            _progress: f64,
            _threshold: f64,
        ) -> Option<<Self::State as State>::Score> {
            Some(state.score(env))
        }

        /// `eval()` 後の変形操作を行う（2-optの区間reverse処理など）
        fn postprocess(&mut self, _env: &Self::Env, _state: &mut Self::State);

        /// `preprocess()` で変形した `state` をロールバックする
        fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State);
    }

    /// 焼きなましの近傍を生成する構造体
    pub trait NeighborGenerator {
        type Env;
        type State: State;

        /// 近傍を生成する
        fn generate(
            &self,
            env: &Self::Env,
            state: &Self::State,
            rng: &mut impl Rng,
        ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>>;
    }

    /// 焼きなましの統計データ
    #[derive(Debug, Clone, Copy)]
    pub struct AnnealingStatistics {
        pub all_iter: usize,
        pub accepted_count: usize,
        pub updated_count: usize,
        pub init_score: i64,
        pub final_score: i64,
    }

    impl AnnealingStatistics {
        fn new(init_score: i64) -> Self {
            Self {
                all_iter: 0,
                accepted_count: 0,
                updated_count: 0,
                init_score,
                final_score: init_score,
            }
        }
    }

    impl Display for AnnealingStatistics {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "===== annealing =====")?;
            writeln!(f, "init score : {}", self.init_score)?;
            writeln!(f, "score      : {}", self.final_score)?;
            writeln!(f, "all iter   : {}", self.all_iter)?;
            writeln!(f, "accepted   : {}", self.accepted_count)?;
            writeln!(f, "updated    : {}", self.updated_count)?;

            Ok(())
        }
    }

    pub struct Annealer<E, S: State<Env = E> + Clone> {
        /// 開始温度
        start_temp: f64,
        /// 終了温度
        end_temp: f64,
        /// 乱数シード
        seed: u128,
        /// 時間計測を行うインターバル
        clock_interval: usize,
        /// イテレーション開始時に呼ばれるコールバック
        callbacks: Vec<Box<dyn Fn(&E, &S, &AnnealingStatistics)>>,
    }

    impl<E, S: State<Env = E> + Clone> Annealer<E, S> {
        pub fn new(
            start_temp: f64,
            end_temp: f64,
            seed: u128,
            clock_interval: usize,
            callbacks: Vec<Box<dyn Fn(&E, &S, &AnnealingStatistics)>>,
        ) -> Self {
            Self {
                start_temp,
                end_temp,
                seed,
                clock_interval,
                callbacks,
            }
        }

        pub fn run<G: NeighborGenerator<Env = E, State = S>>(
            &self,
            env: &E,
            mut state: S,
            neighbor_generator: &G,
            duration_sec: f64,
        ) -> (S, AnnealingStatistics) {
            let mut best_state = state.clone();
            let mut current_score = state.score(&env);
            let mut best_score = current_score.annealing_score(1.0);

            let mut diagnostics = AnnealingStatistics::new(current_score.raw_score());
            let mut rng = Pcg64Mcg::new(self.seed);
            let mut threshold_generator = ThresholdGenerator::new(rng.gen());

            let duration_inv = 1.0 / duration_sec;
            let since = Instant::now();

            let mut progress = 0.0;
            let mut temperature = self.start_temp;

            loop {
                for callback in self.callbacks.iter() {
                    callback(env, &state, &diagnostics);
                }

                diagnostics.all_iter += 1;

                if diagnostics.all_iter % self.clock_interval == 0 {
                    progress = (Instant::now() - since).as_secs_f64() * duration_inv;
                    temperature = f64::powf(self.start_temp, 1.0 - progress)
                        * f64::powf(self.end_temp, progress);

                    if progress >= 1.0 {
                        break;
                    }
                }

                // 変形
                let mut neighbor = neighbor_generator.generate(env, &state, &mut rng);
                neighbor.preprocess(env, &mut state);

                // スコア計算
                let threshold =
                    threshold_generator.next(current_score.annealing_score(progress), temperature);
                let Some(new_score) = neighbor.eval(env, &state, progress, threshold) else {
                    // 明らかに閾値に届かない場合はreject
                    neighbor.rollback(env, &mut state);
                    debug_assert_eq!(state.score(&env), current_score);
                    continue;
                };

                if new_score.annealing_score(progress) >= threshold {
                    // 解の更新
                    neighbor.postprocess(env, &mut state);
                    debug_assert_eq!(state.score(&env), new_score);

                    current_score = new_score;
                    diagnostics.accepted_count += 1;

                    let new_score = current_score.annealing_score(1.0);

                    if best_score < new_score {
                        best_score = new_score;
                        best_state = state.clone();
                        diagnostics.updated_count += 1;
                    }
                } else {
                    neighbor.rollback(env, &mut state);
                    debug_assert_eq!(state.score(&env), current_score);
                }
            }

            diagnostics.final_score = best_state.score(&env).raw_score();

            (best_state, diagnostics)
        }
    }

    /// 焼きなましにおける評価関数の打ち切り基準となる次の閾値を返す構造体
    ///
    /// 参考: [焼きなまし法での評価関数の打ち切り](https://qiita.com/not522/items/cd20b87157d15850d31c)
    struct ThresholdGenerator {
        iter: usize,
        log_randoms: Vec<f64>,
    }

    impl ThresholdGenerator {
        const LEN: usize = 1 << 16;

        fn new(seed: u128) -> Self {
            let mut rng = Pcg64Mcg::new(seed);
            let log_randoms = (0..Self::LEN)
                .map(|_| rng.gen_range(0.0f64..1.0).ln())
                .collect_vec();

            Self {
                iter: 0,
                log_randoms,
            }
        }

        /// 評価関数の打ち切り基準となる次の閾値を返す
        fn next(&mut self, prev_score: f64, temperature: f64) -> f64 {
            let threshold = prev_score + temperature * self.log_randoms[self.iter % Self::LEN];
            self.iter += 1;
            threshold
        }
    }

    #[cfg(test)]
    mod test {
        use itertools::Itertools;
        use rand::Rng;

        use super::{Annealer, Neighbor, Score};

        #[derive(Debug, Clone)]
        struct Input {
            n: usize,
            distances: Vec<Vec<i32>>,
        }

        impl Input {
            fn gen_testcase() -> Self {
                let n = 4;
                let distances = vec![
                    vec![0, 2, 3, 10],
                    vec![2, 0, 1, 3],
                    vec![3, 1, 0, 2],
                    vec![10, 3, 2, 0],
                ];

                Self { n, distances }
            }
        }

        #[derive(Debug, Clone)]
        struct State {
            order: Vec<usize>,
            dist: i32,
        }

        impl State {
            fn new(input: &Input) -> Self {
                let mut order = (0..input.n).collect_vec();
                order.push(0);
                let dist = order
                    .iter()
                    .tuple_windows()
                    .map(|(&prev, &next)| input.distances[prev][next])
                    .sum();

                Self { order, dist }
            }
        }

        impl super::State for State {
            type Env = Input;
            type Score = Dist;

            fn score(&self, _env: &Self::Env) -> Self::Score {
                Dist(self.dist)
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        struct Dist(i32);

        impl Score for Dist {
            fn annealing_score(&self, _progress: f64) -> f64 {
                // 大きい方が良いとするため符号を反転
                -self.0 as f64
            }

            fn raw_score(&self) -> i64 {
                self.0 as i64
            }
        }

        struct TwoOpt {
            begin: usize,
            end: usize,
            new_dist: Option<i32>,
        }

        impl TwoOpt {
            fn new(begin: usize, end: usize) -> Self {
                Self {
                    begin,
                    end,
                    new_dist: None,
                }
            }
        }

        impl Neighbor for TwoOpt {
            type Env = Input;
            type State = State;

            fn preprocess(&mut self, _env: &Self::Env, _state: &mut Self::State) {
                // do nothing
            }

            fn eval(
                &mut self,
                env: &Self::Env,
                state: &Self::State,
                _progress: f64,
                _threshold: f64,
            ) -> Option<<Self::State as super::State>::Score> {
                let v0 = state.order[self.begin - 1];
                let v1 = state.order[self.begin];
                let v2 = state.order[self.end - 1];
                let v3 = state.order[self.end];

                let d00 = env.distances[v0][v1];
                let d01 = env.distances[v0][v2];
                let d10 = env.distances[v2][v3];
                let d11 = env.distances[v1][v3];

                let new_dist = state.dist - d00 - d10 + d01 + d11;
                self.new_dist = Some(new_dist);

                Some(Dist(new_dist))
            }

            fn postprocess(&mut self, _env: &Self::Env, state: &mut Self::State) {
                state.order[self.begin..self.end].reverse();
                state.dist = self
                    .new_dist
                    .expect("postprocess()を呼ぶ前にeval()を呼んでください。");
            }

            fn rollback(&mut self, _env: &Self::Env, _state: &mut Self::State) {
                // do nothing
            }
        }

        struct NeighborGenerator;

        impl super::NeighborGenerator for NeighborGenerator {
            type Env = Input;
            type State = State;

            fn generate(
                &self,
                _env: &Self::Env,
                state: &Self::State,
                rng: &mut impl Rng,
            ) -> Box<dyn Neighbor<Env = Self::Env, State = Self::State>> {
                loop {
                    let begin = rng.gen_range(1..state.order.len());
                    let end = rng.gen_range(1..state.order.len());

                    if begin + 2 <= end {
                        return Box::new(TwoOpt::new(begin, end));
                    }
                }
            }
        }

        #[test]
        fn annealing_tsp_test() {
            let input = Input::gen_testcase();
            let state = State::new(&input);
            let annealer = Annealer::new(1e1, 1e-1, 42, 1000, vec![]);
            let neighbor_generator = NeighborGenerator;

            let (state, diagnostics) = annealer.run(&input, state, &neighbor_generator, 0.1);

            eprintln!("{}", diagnostics);

            eprintln!("score: {}", state.dist);
            eprintln!("state.dist: {:?}", state.order);
            assert_eq!(state.dist, 10);
            assert!(state.order == vec![0, 1, 3, 2, 0] || state.order == vec![0, 2, 3, 1, 0]);
        }
    }
}

#[allow(dead_code)]
mod data_structures {
    use std::slice::Iter;

    /// [0, n) の整数の集合を管理する定数倍が軽いデータ構造
    ///
    /// https://topcoder-tomerun.hatenablog.jp/entry/2021/06/12/134643
    #[derive(Debug, Clone)]
    pub struct IndexSet {
        values: Vec<usize>,
        positions: Vec<Option<usize>>,
    }

    impl IndexSet {
        pub fn new(n: usize) -> Self {
            Self {
                values: vec![],
                positions: vec![None; n],
            }
        }

        pub fn add(&mut self, value: usize) {
            let pos = &mut self.positions[value];

            if pos.is_none() {
                *pos = Some(self.values.len());
                self.values.push(value);
            }
        }

        pub fn remove(&mut self, value: usize) {
            if let Some(index) = self.positions[value] {
                let last = *self.values.last().unwrap();
                self.values[index] = last;
                self.values.pop();
                self.positions[last] = Some(index);
                self.positions[value] = None;
            }
        }

        pub fn contains(&self, value: usize) -> bool {
            self.positions[value].is_some()
        }

        pub fn len(&self) -> usize {
            self.values.len()
        }

        pub fn iter(&self) -> Iter<usize> {
            self.values.iter()
        }

        pub fn as_slice(&self) -> &[usize] {
            &self.values
        }
    }

    /// BFSを繰り返すときに訪問済みかを記録する配列を毎回初期化しなくて良くするアレ
    ///
    /// https://topcoder-tomerun.hatenablog.jp/entry/2022/11/06/145156
    #[derive(Debug, Clone)]
    pub struct FastClearArray {
        values: Vec<u64>,
        gen: u64,
    }

    impl FastClearArray {
        pub fn new(len: usize) -> Self {
            Self {
                values: vec![0; len],
                gen: 1,
            }
        }

        pub fn clear(&mut self) {
            self.gen += 1;
        }

        pub fn set_true(&mut self, index: usize) {
            self.values[index] = self.gen;
        }

        pub fn get(&self, index: usize) -> bool {
            self.values[index] == self.gen
        }

        pub fn len(&self) -> usize {
            self.values.len()
        }
    }

    #[cfg(test)]
    mod test {
        use super::{FastClearArray, IndexSet};
        use itertools::Itertools;

        #[test]
        fn index_set() {
            let mut set = IndexSet::new(10);
            set.add(1);
            set.add(5);
            set.add(2);
            assert_eq!(3, set.len());
            assert!(set.contains(1));
            assert!(!set.contains(0));
            assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1, 2, 5]);
            assert_eq!(
                set.as_slice().iter().copied().sorted().collect_vec(),
                vec![1, 2, 5]
            );

            set.add(1);
            assert_eq!(3, set.len());
            assert!(set.contains(1));
            assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1, 2, 5]);

            set.remove(5);
            set.remove(2);
            assert_eq!(1, set.len());
            assert!(set.contains(1));
            assert!(!set.contains(5));
            assert!(!set.contains(2));
            assert_eq!(set.iter().copied().sorted().collect_vec(), vec![1]);

            set.remove(1);
            set.remove(2);
            assert_eq!(0, set.len());
            assert!(!set.contains(1));
            assert_eq!(set.iter().copied().sorted().collect_vec(), vec![]);
        }

        #[test]
        fn fast_clear_array() {
            let mut array = FastClearArray::new(5);
            assert_eq!(array.get(0), false);

            array.set_true(0);
            assert_eq!(array.get(0), true);
            assert_eq!(array.get(1), false);

            array.clear();
            assert_eq!(array.get(0), false);

            array.set_true(0);
            assert_eq!(array.get(0), true);
        }
    }
}

#[allow(dead_code)]
mod grid {
    use std::ops::{Add, AddAssign, Index, IndexMut};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coord {
        row: u8,
        col: u8,
    }

    impl Coord {
        pub const fn new(row: usize, col: usize) -> Self {
            Self {
                row: row as u8,
                col: col as u8,
            }
        }

        pub const fn row(&self) -> usize {
            self.row as usize
        }

        pub const fn col(&self) -> usize {
            self.col as usize
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size as u8 && self.col < size as u8
        }

        pub const fn to_index(&self, size: usize) -> usize {
            self.row as usize * size + self.col as usize
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: u8, x1: u8) -> usize {
            x0.abs_diff(x1) as usize
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordDiff {
        dr: i8,
        dc: i8,
    }

    impl CoordDiff {
        pub const fn new(dr: i32, dc: i32) -> Self {
            Self {
                dr: dr as i8,
                dc: dc as i8,
            }
        }

        pub const fn invert(&self) -> Self {
            Self {
                dr: -self.dr,
                dc: -self.dc,
            }
        }

        pub const fn dr(&self) -> i32 {
            self.dr as i32
        }

        pub const fn dc(&self) -> i32 {
            self.dc as i32
        }
    }

    impl Add<CoordDiff> for Coord {
        type Output = Coord;

        fn add(self, rhs: CoordDiff) -> Self::Output {
            Coord {
                row: self.row.wrapping_add(rhs.dr as u8),
                col: self.col.wrapping_add(rhs.dc as u8),
            }
        }
    }

    impl AddAssign<CoordDiff> for Coord {
        fn add_assign(&mut self, rhs: CoordDiff) {
            self.row = self.row.wrapping_add(rhs.dr as u8);
            self.col = self.col.wrapping_add(rhs.dc as u8);
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
    pub struct CoordIndex(pub usize);

    #[allow(dead_code)]
    impl CoordIndex {
        pub const fn new(index: usize) -> Self {
            Self(index)
        }

        pub const fn to_coord(&self, n: usize) -> Coord {
            Coord::new(self.0 / n, self.0 % n)
        }
    }

    pub const ADJACENTS: [CoordDiff; 4] = [
        CoordDiff::new(-1, 0),
        CoordDiff::new(0, 1),
        CoordDiff::new(1, 0),
        CoordDiff::new(0, -1),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        size: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, size: usize) -> Self {
            debug_assert!(size * size == map.len());
            Self { size, map }
        }
    }

    impl<T: Default + Clone> Map2d<T> {
        pub fn with_default(size: usize) -> Self {
            let map = vec![T::default(); size * size];
            Self { size, map }
        }
    }

    impl<T> Index<Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<&Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<&Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &self.map[begin..end]
        }
    }

    impl<T> IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &mut self.map[begin..end]
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConstMap2d<T, const N: usize> {
        map: Vec<T>,
    }

    impl<T, const N: usize> ConstMap2d<T, N> {
        pub fn new(map: Vec<T>) -> Self {
            assert_eq!(map.len(), N * N);
            Self { map }
        }
    }

    impl<T: Default + Clone, const N: usize> ConstMap2d<T, N> {
        pub fn with_default() -> Self {
            let map = vec![T::default(); N * N];
            Self { map }
        }
    }

    impl<T, const N: usize> Index<Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<&Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<&Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<CoordIndex> for ConstMap2d<T, N> {
        type Output = T;

        fn index(&self, index: CoordIndex) -> &Self::Output {
            &self.map[index.0]
        }
    }

    impl<T, const N: usize> IndexMut<CoordIndex> for ConstMap2d<T, N> {
        fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
            &mut self.map[index.0]
        }
    }

    impl<T, const N: usize> Index<usize> for ConstMap2d<T, N> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * N;
            let end = begin + N;
            &self.map[begin..end]
        }
    }

    impl<T, const N: usize> IndexMut<usize> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * N;
            let end = begin + N;
            &mut self.map[begin..end]
        }
    }

    #[cfg(test)]
    mod test {
        use super::{ConstMap2d, Coord, CoordDiff, Map2d};

        #[test]
        fn coord_add() {
            let c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            let actual = c + d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, actual);
        }

        #[test]
        fn coord_add_assign() {
            let mut c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            c += d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, c);
        }

        #[test]
        fn map_new() {
            let map = Map2d::new(vec![0, 1, 2, 3], 2);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn map_default() {
            let map = Map2d::with_default(2);
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_new() {
            let map = ConstMap2d::<_, 2>::new(vec![0, 1, 2, 3]);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_default() {
            let map = ConstMap2d::<_, 2>::with_default();
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }
    }
}
