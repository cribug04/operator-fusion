//! `cargo run -p resize-plan`：搭一张「输入图 ⇒ 双线性 ×2 ⇒ 输出」的图 ⇒ 用 resize_probe 的引擎表编排计划 ⇒ 打印作业 ⇒ 示意地址 bind。
//! 只证明「Resize 已注册进 calDNN 图 / 引擎表、能被图调用并编排」；⛔ 没有发射、没有上板。
use caldnn::{Arg, Elem, FirstFit, Graph, Plan, Uid, VariantPack};
use resize_probe::engines::ENGINES;

fn main() {
    let mut g = Graph::default();
    let x = g.input(Elem::Bf16, &[32, 32, 16]);
    let y = g.upsample_bilinear2(x);
    let y = g.output(y);
    let plan = Plan::new(&g, ENGINES, &FirstFit::default()).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1)
    });
    let [ddr, sram] = plan.workspace();
    println!("{} 个张量、{} 个运算 ⇒ {} 个作业，工作区 DDR {ddr} 字节、SRAM {sram} 字节", g.tensors().len(), g.ops().len(), plan.jobs().len());
    for j in plan.jobs() {
        let args: Vec<String> = j.args.iter().map(|a| if let Arg::Ptr(u) = a { format!("%{u}") } else { format!("{a:?}") }).collect();
        println!("  {}({})  arg_kinds = {:?}", j.kernel, args.join(", "), j.arg_kinds);
    }
    let ptrs = [(x, 0x1_0000_0000u64), (y, 0x1_0001_0000u64)];
    let jobs = plan.bind(&VariantPack { ptrs: &ptrs, workspace: 0x1_0010_0000, sram: 0x0800_0000 }).unwrap_or_else(|e| {
        eprintln!("没给地址：{e:?}");
        std::process::exit(1)
    });
    for j in &jobs {
        println!("bind ⇒ {}({:?})", j.kernel, j.args);
    }
    let _: Uid = y;
}

#[cfg(test)]
mod tests {
    use super::*;
    use caldnn::{Interp, Op, ResizeDesc};

    fn plan_for(dims: &[u32]) -> Result<Plan<'static>, caldnn::Unsupported> {
        let mut g = Graph::default();
        let x = g.input(Elem::Bf16, dims);
        let y = g.upsample_bilinear2(x);
        g.output(y);
        Plan::new(&g, ENGINES, &FirstFit::default())
    }

    #[test]
    fn graph_records_resize_op_and_output_dims() {
        let mut g = Graph::default();
        let x = g.input(Elem::Bf16, &[32, 32, 16]);
        let y = g.upsample_bilinear2(x);
        assert_eq!(g[y].dims, vec![64, 64, 16]);
        assert_eq!(g[y].elem, Elem::Bf16);
        assert!(g[y].virt, "没标 output 前是虚拟张量");
        let g2 = g.clone();
        g.output(y);
        assert!(!g[y].virt);
        let expect = ResizeDesc { h: 32, w: 32, c: 16, scale: 2, mode: Interp::Bilinear };
        assert_eq!(g2.ops(), &[Op::Resize(expect, [x, y])]);
    }

    #[test]
    fn engine_table_matches_specialized_shape() {
        let plan = plan_for(&[32, 32, 16]).expect("32×32×16 在引擎表里");
        assert_eq!(plan.jobs().len(), 1);
        let j = &plan.jobs()[0];
        assert_eq!(j.kernel, "resize_bilinear2");
        assert_eq!(j.args, vec![Arg::Ptr(0), Arg::Ptr(1)], "参数是张量号 x=%0、y=%1");
        assert_eq!(plan.workspace(), [0, 0], "输出已标 output，没有虚拟张量要分工作区");
    }

    #[test]
    fn other_shapes_are_unsupported() {
        for dims in [&[16u32, 16, 16][..], &[32, 32, 32], &[64, 64, 16]] {
            let e = plan_for(dims).err().expect("不在表里的形状要报 Unsupported");
            assert_eq!(e.at, 0);
            assert!(matches!(e.op, Op::Resize(..)));
        }
    }

    #[test]
    fn bind_replaces_uids_with_addresses() {
        let mut g = Graph::default();
        let x = g.input(Elem::Bf16, &[32, 32, 16]);
        let y = g.upsample_bilinear2(x);
        let y = g.output(y);
        let plan = Plan::new(&g, ENGINES, &FirstFit::default()).unwrap();
        let ptrs = [(x, 0x1000u64), (y, 0x2000u64)];
        let jobs = plan.bind(&VariantPack { ptrs: &ptrs, workspace: 0, sram: 0 }).unwrap();
        assert_eq!(jobs[0].args, vec![Arg::Ptr(0x1000), Arg::Ptr(0x2000)]);
        assert!(matches!(plan.bind(&VariantPack { ptrs: &ptrs[..1], workspace: 0, sram: 0 }), Err(caldnn::Unbound(u)) if u == y));
    }

    #[test]
    fn intermediate_resize_gets_workspace() {
        // 图里连用两次 ×2：中间结果是虚拟张量，计划要给它分工作区（64·64·16·2 = 131072 B，512 B 对齐）
        let mut g = Graph::default();
        let x = g.input(Elem::Bf16, &[32, 32, 16]);
        let mid = g.upsample_bilinear2(x);
        let y = g.upsample_bilinear2(mid);
        g.output(y);
        // 第二个 ×2 是 64×64×16，表里没有 ⇒ Unsupported at 1；但工作区的逻辑在 Plan::new 里，先确认报错位置
        let e = Plan::new(&g, ENGINES, &FirstFit::default()).err().unwrap();
        assert_eq!(e.at, 1);
    }
}
