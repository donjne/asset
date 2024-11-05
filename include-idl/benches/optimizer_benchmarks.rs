use criterion::{black_box, criterion_group, criterion_main, Criterion};
use serde_json::json;
use solana_include_idl::just_optimize::{IdlOptimizer, OptimizationConfig, OptimizationType};
use std::time::Duration;

/// Generate test IDL with configurable size parameters
fn generate_large_idl(type_count: usize, field_count: usize, doc_size: usize) -> serde_json::Value {
    println!("\nGenerating test IDL with:");
    println!("- {} types", type_count);
    println!("- {} fields per type", field_count);
    println!("- {} bytes of documentation", doc_size);

    let mut types = Vec::new();
    let doc_string = "Documentation text. ".repeat(doc_size / 20);
    
    for i in 0..type_count {
        let mut fields = Vec::new();
        for j in 0..field_count {
            fields.push(json!({
                "name": format!("field_{}", j),
                "type": "string",
                "docs": [doc_string.clone()],
                "internal_metadata": {
                    "created_at": "timestamp",
                    "debug_info": "some debug data",
                    "validation_rules": ["rule1", "rule2"]
                }
            }));
        }
        
        // Create some duplicate types to test deduplication
        let type_name = if i % 2 == 0 {
            format!("Type_{}", i / 2)
        } else {
            format!("DuplicateType_{}", i / 2)
        };
        
        types.push(json!({
            "name": type_name,
            "type": "struct",
            "fields": fields,
            "docs": [doc_string.clone()],
            "description": doc_string.clone(),
            "internal_id": format!("internal_{}", i),
            "debug_mode": true
        }));
    }
    
    json!({
        "version": "1.0.0",
        "name": "benchmark_program",
        "docs": ["Main documentation"],
        "types": types,
        "instructions": [
            {
                "name": "initialize",
                "docs": ["Instruction documentation"],
                "accounts": [],
                "args": []
            }
        ]
    })
}

fn benchmark_size_reduction(c: &mut Criterion) {
    println!("\n=== Starting Size Optimization Benchmarks ===");
    let mut group = c.benchmark_group("Size Optimization");
    
    // Configure benchmark parameters
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);
    
    // Test different IDL sizes
    let sizes = [
        (5, 5, 100),    // Small IDL
        (20, 10, 500),   // Medium IDL
        (50, 20, 1000),  // Large IDL
    ];
    
    for size in sizes {
        let (type_count, field_count, doc_size) = size;
        println!("\n--- Testing IDL Size Configuration ---");
        println!("Types: {}, Fields: {}, Doc Size: {}", type_count, field_count, doc_size);
        
        let idl = generate_large_idl(type_count, field_count, doc_size);
        let original_size = serde_json::to_string(&idl).unwrap().len();
        println!("Original IDL size: {} bytes", original_size);
        
        // Benchmark with different optimization combinations
        let configs = vec![
            ("docs_only", {
                let mut config = OptimizationConfig::default();
                config.enable(OptimizationType::DocumentationRemoval);
                config
            }),
            ("full_opt", {
                let mut config = OptimizationConfig::default();
                config.enable(OptimizationType::DocumentationRemoval)
                    .enable(OptimizationType::TypeDeduplication)
                    .enable(OptimizationType::StringMinification)
                    .enable(OptimizationType::CustomFieldRemoval)
                    .enable(OptimizationType::Compression);
                config
            }),
            ("no_compression", {
                let mut config = OptimizationConfig::default();
                config.enable(OptimizationType::DocumentationRemoval)
                    .enable(OptimizationType::TypeDeduplication)
                    .enable(OptimizationType::StringMinification)
                    .enable(OptimizationType::CustomFieldRemoval);
                config
            }),
        ];
        
        for (name, config) in configs {
            println!("\nRunning {} optimization:", name);
            
            // Run initial optimization to show immediate results
            let mut optimizer = IdlOptimizer::new(config.clone());
            let (optimized, metrics) = optimizer.optimize(idl.clone()).unwrap();
            let optimized_size = serde_json::to_string(&optimized).unwrap().len();
            
            println!("Results:");
            println!("- Final size: {} bytes", optimized_size);
            println!("- Reduction: {} bytes ({:.2}%)",
                original_size - optimized_size,
                (original_size - optimized_size) as f64 / original_size as f64 * 100.0
            );
            println!("- Items affected: {:?}", metrics.items_affected);
            println!("- Processing time: {:?}", metrics.processing_time);
            
            // Run the actual benchmark
            group.bench_function(
                format!("{}_{}_types_{}_fields", name, type_count, field_count),
                |b| b.iter(|| {
                    let mut optimizer = IdlOptimizer::new(config.clone());
                    let (optimized, metrics) = optimizer.optimize(black_box(idl.clone())).unwrap();
                    let optimized_size = serde_json::to_string(&optimized).unwrap().len();
                    (optimized_size, metrics)
                })
            );
        }
    }
    
    group.finish();
    println!("\n=== Size Optimization Benchmarks Completed ===");
}

fn analyze_optimization_results(c: &mut Criterion) {
    println!("\n=== Starting Optimization Analysis ===");
    
    // Create a large test IDL
    let idl = generate_large_idl(30, 15, 750);
    let original_size = serde_json::to_string(&idl).unwrap().len();
    
    println!("\nOptimization Analysis:");
    println!("Original Size: {} bytes", original_size);
    
    // Test each optimization individually
    let optimizations = vec![
        (OptimizationType::DocumentationRemoval, "Documentation Removal"),
        (OptimizationType::TypeDeduplication, "Type Deduplication"),
        (OptimizationType::StringMinification, "String Minification"),
        (OptimizationType::CustomFieldRemoval, "Custom Field Removal"),
        (OptimizationType::Compression, "Compression"),
    ];
    
    // First loop - borrow the vector
    for (opt_type, name) in &optimizations {
        println!("\nTesting: {}", name);
        let mut config = OptimizationConfig::default();
        config.enable(*opt_type);  // Dereference opt_type since it's now a reference
        
        let mut optimizer = IdlOptimizer::new(config);
        let (optimized, metrics) = optimizer.optimize(idl.clone()).unwrap();
        let optimized_size = serde_json::to_string(&optimized).unwrap().len();
        
        println!("Results:");
        println!("- Size Reduction: {} bytes ({:.2}%)",
            original_size - optimized_size,
            (original_size - optimized_size) as f64 / original_size as f64 * 100.0
        );
        println!("- Items Affected: {}", 
            metrics.items_affected.get(opt_type).unwrap_or(&0)  // Use opt_type directly as it's now a reference
        );
        println!("- Processing Time: {:?}", 
            metrics.processing_time.get(opt_type).unwrap_or(&Duration::new(0, 0))
        );
    }
    
    // Test all optimizations together
    println!("\nTesting all optimizations combined:");
    let mut config = OptimizationConfig::default();
    // Second loop - still using the borrowed vector
    for (opt_type, _) in &optimizations {
        config.enable(*opt_type);  // Dereference opt_type
    }
    
    let mut optimizer = IdlOptimizer::new(config.clone());
    let (optimized, metrics) = optimizer.optimize(idl.clone()).unwrap();
    let optimized_size = serde_json::to_string(&optimized).unwrap().len();
    
    println!("\nFinal Results:");
    println!("- Final Size: {} bytes", optimized_size);
    println!("- Total Reduction: {} bytes ({:.2}%)",
        original_size - optimized_size,
        (original_size - optimized_size) as f64 / original_size as f64 * 100.0
    );
    println!("- Metrics: {:?}", metrics);
    
    // Add to Criterion benchmarks
    let mut group = c.benchmark_group("Full Optimization Analysis");
    group.bench_function("all_optimizations", |b| b.iter(|| {
        let mut optimizer = IdlOptimizer::new(config.clone());
        optimizer.optimize(black_box(idl.clone())).unwrap()
    }));
    group.finish();
    
    println!("\n=== Optimization Analysis Completed ===");
}

criterion_group!(benches, benchmark_size_reduction, analyze_optimization_results);
criterion_main!(benches);