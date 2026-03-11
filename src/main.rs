use zk_torch::{init_from_yaml};
use zk_torch::util::zktorch_kernel;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 { panic!("Usage: cargo run -- <config file>"); }
    let yaml = std::fs::read_to_string(&args[1]).expect("read config");
    init_from_yaml(&yaml).expect("init config");
    zktorch_kernel(); 
}
