use io_uring::{opcode, types, IoUring};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::time::Instant;

// =====================================================================
// THESIS EXPERIMENT: RUST io_uring ZERO-COPY DMA (Academic Edition)
// =====================================================================
// Outputs to `io_uring_metrics.csv` with multiple trials for statistical
// rigor. Tracks raw physical IO bandwidth limits.

const BLOCK_SIZE: usize = 128 * 1024 * 1024;
const NUM_BLOCKS: usize = 16;
const QUEUE_DEPTHS: [u32; 2] = [1, 4];
const TRIALS: usize = 3;

fn main() {
    let filepath = "raw_layer_fp16.bin";
    let mut csv_file = File::create("io_uring_metrics.csv").expect("Failed to create CSV");
    writeln!(csv_file, "QueueDepth,Trial,Data_MB,Time_s,Bandwidth_MBs").unwrap();

    println!("========================================================");
    println!(" THESIS: RAW RUST io_uring BANDWIDTH (Multi-Trial & QD)");
    println!("========================================================");

    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(filepath)
    {
        Ok(f) => f,
        Err(_) => {
            println!("Error: O_DIRECT failed. Ensure you are on ext4/xfs.");
            return;
        }
    };

    let fd = file.as_raw_fd();

    let mut buffers: Vec<*mut libc::c_void> = Vec::with_capacity(NUM_BLOCKS);
    for _ in 0..NUM_BLOCKS {
        let mut ptr: *mut libc::c_void = std::ptr::null_mut();
        unsafe {
            if libc::posix_memalign(&mut ptr, 4096, BLOCK_SIZE) != 0 {
                panic!("Memory alignment failed");
            }
        }
        buffers.push(ptr);
    }

    for &queue_depth in &QUEUE_DEPTHS {
        println!("\n--- Testing Queue Depth: {} ---", queue_depth);
        let mut ring = IoUring::new(queue_depth).expect("Failed to initialize io_uring");

        for trial in 1..=TRIALS {
            let start_time = Instant::now();
            let mut blocks_submitted = 0;
            let mut blocks_completed = 0;
            let mut offset: u64 = 0;

            while blocks_submitted < queue_depth as usize && blocks_submitted < NUM_BLOCKS {
                let read_e = opcode::Read::new(
                    types::Fd(fd),
                    buffers[blocks_submitted] as *mut u8,
                    BLOCK_SIZE as u32,
                )
                .offset(offset)
                .build()
                .user_data(blocks_submitted as u64);

                unsafe {
                    ring.submission().push(&read_e).unwrap();
                }
                offset += BLOCK_SIZE as u64;
                blocks_submitted += 1;
            }
            ring.submit().unwrap();

            while blocks_completed < NUM_BLOCKS {
                ring.submit_and_wait(1).unwrap();
                let mut cq = ring.completion();

                while let Some(cqe) = cq.next() {
                    if cqe.result() < 0 {
                        panic!("Read error: {}", cqe.result());
                    }
                    blocks_completed += 1;

                    if blocks_submitted < NUM_BLOCKS {
                        let read_e = opcode::Read::new(
                            types::Fd(fd),
                            buffers[blocks_submitted] as *mut u8,
                            BLOCK_SIZE as u32,
                        )
                        .offset(offset)
                        .build()
                        .user_data(blocks_submitted as u64);

                        unsafe {
                            ring.submission().push(&read_e).unwrap();
                        }
                        offset += BLOCK_SIZE as u64;
                        blocks_submitted += 1;
                    }
                }
            }

            let duration = start_time.elapsed().as_secs_f64();
            let total_mb = (NUM_BLOCKS * BLOCK_SIZE) as f64 / (1024.0 * 1024.0);
            let bandwidth = total_mb / duration;

            println!(" Trial {}: BW = {:.2} MB/s", trial, bandwidth);
            writeln!(
                csv_file,
                "{}, {}, {:.2}, {:.4}, {:.2}",
                queue_depth, trial, total_mb, duration, bandwidth
            )
            .unwrap();
        }
    }

    println!("========================================================");
    println!("[+] Academic data saved to 'io_uring_metrics.csv' for charting.");

    for ptr in buffers {
        unsafe { libc::free(ptr) };
    }
}
