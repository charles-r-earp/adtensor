use rspirv;
use spirv_headers as spirv;

use rspirv::binary::Assemble;
use rspirv::binary::Disassemble;

fn main() {
  // Building
  let mut b = rspirv::mr::Builder::new();
  b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
  let void = b.type_void();
  let voidf = b.type_function(void, vec![void]);
  b.begin_function(void,
                   None,
                   spirv::FunctionControl::DONT_INLINE |
                    spirv::FunctionControl::CONST,
                   voidf)
   .unwrap();
  b.begin_basic_block(None).unwrap();
  b.ret().unwrap();
  b.end_function().unwrap();
  let module = b.module();

  // Assembling
  let code = module.assemble();
  assert!(code.len() > 20);  // Module header contains 5 words
  assert_eq!(spirv::MAGIC_NUMBER, code[0]);

  // Parsing
  let mut loader = rspirv::mr::Loader::new();
  rspirv::binary::parse_words(&code, &mut loader).unwrap();
  let module = loader.module();
  println!("{:?}", module);
  println!("{}", module.disassemble());
  /*
  // Disassembling
  assert_eq!(module.disassemble(),
             "; SPIR-V\n\
              ; Version: 1.3\n\
              ; Generator: rspirv\n\
              ; Bound: 5\n\
              OpMemoryModel Logical GLSL450\n\
              %1 = OpTypeVoid\n\
              %2 = OpTypeFunction %1 %1\n\
              %3 = OpFunction  %1  DontInline|Const %2\n\
              %4 = OpLabel\n\
              OpReturn\n\
              OpFunctionEnd");*/
}
