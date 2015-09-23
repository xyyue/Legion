/* Copyright 2015 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include "circuit.h"
#include <x86intrin.h>

using namespace LegionRuntime::Accessor;

const float AccumulateCharge::identity = 0.0f;

template <>
void AccumulateCharge::apply<true>(LHS &lhs, RHS rhs) 
{
  lhs += rhs;
}

template<>
void AccumulateCharge::apply<false>(LHS &lhs, RHS rhs)
{
  int *target = (int *)&lhs;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

template <>
void AccumulateCharge::fold<true>(RHS &rhs1, RHS rhs2) 
{
  rhs1 += rhs2;
}

template<>
void AccumulateCharge::fold<false>(RHS &rhs1, RHS rhs2)
{
  int *target = (int *)&rhs1;
  union { int as_int; float as_float; } oldval, newval;
  do {
    oldval.as_int = *target;
    newval.as_float = oldval.as_float + rhs2;
  } while (!__sync_bool_compare_and_swap(target, oldval.as_int, newval.as_int));
}

CalcNewCurrentsTask::CalcNewCurrentsTask(LogicalPartition lp_pvt_wires,
                                         LogicalPartition lp_pvt_nodes,
                                         LogicalPartition lp_shr_nodes,
                                         LogicalPartition lp_ghost_nodes,
                                         LogicalPartition lp_inside_nodes,
                                         LogicalRegion lr_all_wires,
                                         LogicalRegion lr_all_nodes,
                                         const Domain &launch_domain,
                                         const ArgumentMap &arg_map)
 : IndexLauncher(CalcNewCurrentsTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, CalcNewCurrentsTask::MAPPER_ID)
{
  
  RegionRequirement rr_wires(lp_pvt_wires, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_wires); // Second region
  rr_wires.add_field(FID_IN_PTR);
  rr_wires.add_field(FID_OUT_PTR);
  rr_wires.add_field(FID_IN_LOC);
  rr_wires.add_field(FID_OUT_LOC);
  rr_wires.add_field(FID_WIRE_VALUE);
  rr_wires.add_field(FID_PIECE_NUM1);
  rr_wires.add_field(FID_PIECE_NUM2);
  add_region_requirement(rr_wires);

  RegionRequirement rr_private(lp_pvt_nodes, 0/*identity*/,
                               READ_ONLY, EXCLUSIVE, lr_all_nodes); // Third Region
  rr_private.add_field(FID_NODE_VALUE);
  rr_private.add_field(FID_NODE_OFFSET);
  add_region_requirement(rr_private);

  RegionRequirement rr_shared(lp_shr_nodes, 0/*identity*/,
                              READ_ONLY, EXCLUSIVE, lr_all_nodes);// 4th Region
  rr_shared.add_field(FID_NODE_VALUE);
  rr_shared.add_field(FID_NODE_OFFSET);
  add_region_requirement(rr_shared);

  RegionRequirement rr_ghost(lp_ghost_nodes, 0/*identity*/,
                             READ_ONLY, EXCLUSIVE, lr_all_nodes); // 5th Region
  rr_ghost.add_field(FID_NODE_VALUE);
  add_region_requirement(rr_ghost);


  RegionRequirement rr_private_result(lp_pvt_nodes, 0/*identity*/,  // add the result field 
                             READ_WRITE, EXCLUSIVE, lr_all_nodes);  // 6th Region
  rr_private_result.add_field(FID_NODE_RESULT);
  add_region_requirement(rr_private_result);


  RegionRequirement rr_shared_result(lp_shr_nodes, 0/*identity*/,
                             READ_WRITE, EXCLUSIVE, lr_all_nodes);  // 7th Region
  rr_shared_result.add_field(FID_NODE_RESULT);
  add_region_requirement(rr_shared_result);


  RegionRequirement rr_inside(lp_inside_nodes, 0/*identity*/,
                               READ_WRITE, EXCLUSIVE, lr_all_nodes); // 8th  Region
  rr_inside.add_field(FID_NODE_OFFSET);
  rr_inside.add_field(FID_NODE_RESULT);
  add_region_requirement(rr_inside);
  //RegionRequirement rr_ghost_result(lp_ghost_nodes, 0/*identity*/,
  //                           READ_WRITE, EXCLUSIVE, lr_all_nodes);  // 8th Region
  //rr_shared_result.add_field(FID_NODE_RESULT);
  //add_region_requirement(rr_ghost_result);
}

/*static*/ const char * const CalcNewCurrentsTask::TASK_NAME = "calc_new_currents";

bool CalcNewCurrentsTask::launch_check_fields(Context ctx, HighLevelRuntime *runtime)
{
  const RegionRequirement &req = region_requirements[0];
  bool success = true;
  for (int i = 0; i < WIRE_SEGMENTS; i++)
  {
    CheckTask launcher(req.partition, req.parent, FID_CURRENT+i, launch_domain, argument_map); 
    success = launcher.dispatch(ctx, runtime, success); 
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    CheckTask launcher(req.partition, req.parent, FID_WIRE_VOLTAGE+i, launch_domain, argument_map);
    success = launcher.dispatch(ctx, runtime, success);
  }
  return success;
}


template<typename AT>
static inline double get_node_value(Context ctx, HighLevelRuntime* rt,
                                    const RegionAccessor<AT,double> &priv,
                                    const RegionAccessor<AT,double> &shr,
                                    const RegionAccessor<AT,double> &ghost,
                                    LogicalRegion &pvt_region,
                                    LogicalRegion &shr_region,
                                    LogicalRegion &ghost_region,
                                    PointerLocation loc, ptr_t ptr)
{
  if (rt->safe_cast(ctx, ptr, pvt_region))
    return priv.read(ptr);
  if (rt->safe_cast(ctx, ptr, shr_region))
    return shr.read(ptr);
  if (rt->safe_cast(ctx, ptr, ghost_region))
    return ghost.read(ptr);
    assert(false);
  return 0.f;
}
 
template<typename AT>
void process_result(Context ctx, HighLevelRuntime* rt, 
                    const RegionAccessor<AT,double> &priv,
                    const RegionAccessor<AT,double> &shr,
                    LogicalRegion &pvt_region,
                    LogicalRegion &shr_region,
                    ptr_t ptr,
                    double wire_value,
                    double node_value)
{
  double result;
  if (rt->safe_cast(ctx, ptr, pvt_region))
  {
    result = priv.read(ptr);
    //printf("Previous: %f\n", result);
    //printf("New: Previous + %f * %f = %f\n", wire_value, node_value, result + wire_value * node_value);
    priv.write(ptr, result + wire_value * node_value);
  }
  else if (rt->safe_cast(ctx, ptr, shr_region))
  {
    result = shr.read(ptr);
    //printf("Previous: %f\n", result);
    //printf("New: Previous + %f * %f = %f\n", wire_value, node_value, result + wire_value * node_value);
    shr.write(ptr, result + wire_value * node_value);
  }
  //double in_result = get_node_result(ctx, rt, fa_pvt_result, fa_shr_result, pvt_region, shr_region, in_loc, in_ptr);
  //write_node_value(fa_pvt_result, fa_shr_result, in_loc,
  //                in_ptr, in_result + wire_value * out_node_value);
}

//template<typename AT>
//static inline int get_node_result(const RegionAccessor<AT,double> &priv,
//                                    const RegionAccessor<AT,double> &shr,
//                                    PointerLocation loc, ptr_t ptr)
//{
//  switch (loc)
//  {
//    case PRIVATE_PTR:
//      return priv.read(ptr);
//    case SHARED_PTR:
//      return shr.read(ptr);
//    default:
//      assert(false);
//  }
//  return 0.f;
//}


//template<typename AT>
//static inline void write_node_value(const RegionAccessor<AT,double> &priv,
//                                    const RegionAccessor<AT,double> &shr,
//                                    PointerLocation loc, ptr_t ptr, double val)
//{
//  switch (loc)
//  {
//    case PRIVATE_PTR:
//      priv.write(ptr, val);
//      break;
//    case SHARED_PTR:
//      shr.write(ptr, val);
//      break;
//    default:
//     assert(false);
//  }
//  return;
//}


/*static*/
void CalcNewCurrentsTask::cpu_base_impl(const CircuitPiece &p,
                                        const std::vector<PhysicalRegion> &regions,
                                        Context ctx, HighLevelRuntime* rt)
{
  RegionAccessor<AccessorType::Generic, ptr_t> fa_in_ptr = 
    regions[0].get_field_accessor(FID_IN_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_out_ptr = 
    regions[0].get_field_accessor(FID_OUT_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_in_loc = 
    regions[0].get_field_accessor(FID_IN_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_out_loc = 
    regions[0].get_field_accessor(FID_OUT_LOC).typeify<PointerLocation>();

  RegionAccessor<AccessorType::Generic, double> fa_wire_value =  // newly added
    regions[0].get_field_accessor(FID_WIRE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num1 =
    regions[0].get_field_accessor(FID_PIECE_NUM1).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num2 =
    regions[0].get_field_accessor(FID_PIECE_NUM2).typeify<int>();

  RegionAccessor<AccessorType::Generic, double> fa_pvt_value = 
    regions[1].get_field_accessor(FID_NODE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_shr_value = 
    regions[2].get_field_accessor(FID_NODE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_ghost_value = 
    regions[3].get_field_accessor(FID_NODE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_pvt_result = 
    regions[4].get_field_accessor(FID_NODE_RESULT).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_shr_result = 
    regions[5].get_field_accessor(FID_NODE_RESULT).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_node_offset = 
    regions[6].get_field_accessor(FID_NODE_OFFSET).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_node_result = 
    regions[6].get_field_accessor(FID_NODE_RESULT).typeify<double>();

  LogicalRegion pvt_region = regions[1].get_logical_region();
  LogicalRegion shr_region = regions[2].get_logical_region();
  LogicalRegion ghost_region = regions[3].get_logical_region();


  LegionRuntime::HighLevel::IndexIterator itr(rt, ctx, p.pvt_wires);
  int piece_num = p.piece_num;                   // newly added

  //printf("before the while loop from piece %d\n", piece_num);
  while (itr.has_next())
  {
    ptr_t wire_ptr = itr.next();

    // Pin the outer voltages to the node voltages
    ptr_t in_ptr = fa_in_ptr.read(wire_ptr);
    PointerLocation in_loc = fa_in_loc.read(wire_ptr);

    ptr_t out_ptr = fa_out_ptr.read(wire_ptr);
    PointerLocation out_loc = fa_out_loc.read(wire_ptr);
    
    /********************************newly added**************************************/

    int m1 = fa_piece_num1.read(wire_ptr);
    int m2 = fa_piece_num2.read(wire_ptr);

    double in_node_value = get_node_value(ctx, rt, fa_pvt_value, fa_shr_value, fa_ghost_value, 
                  pvt_region, shr_region, ghost_region, in_loc, in_ptr); 
    double out_node_value = get_node_value(ctx, rt, fa_pvt_value, fa_shr_value, fa_ghost_value, 
                  pvt_region, shr_region, ghost_region, out_loc, out_ptr); 
    double wire_value = fa_wire_value.read(wire_ptr);
    //printf("******************************************************************\n");
    //printf("The value of the wire is: %f\n", wire_value);
    //printf("the two values of the wire is: (%f, %f)\n", in_node_value, out_node_value);

    if (m1 == piece_num)
    {
      //printf("The 1st node value is: %f\n", in_node_value);
      process_result(ctx, rt, fa_pvt_result, fa_shr_result, pvt_region, shr_region, in_ptr, wire_value, out_node_value);
      //double in_result = get_node_result(ctx, rt, fa_pvt_result, fa_shr_result, pvt_region, shr_region, in_loc, in_ptr);
      //write_node_value(fa_pvt_result, fa_shr_result, in_loc,
      //                in_ptr, in_result + wire_value * out_node_value);
    }
    if (m2 == piece_num && in_ptr != out_ptr)
    {
      //printf("The 2nd node value is: %f\n", out_node_value);
      process_result(ctx, rt, fa_pvt_result, fa_shr_result, pvt_region, shr_region, out_ptr, wire_value, in_node_value);
      //double out_result = get_node_result(fa_pvt_result, fa_shr_result, out_loc, out_ptr);
      //write_node_value(fa_pvt_result, fa_shr_result, out_loc,
      //                out_ptr, out_result + wire_value * in_node_value);
    }

    /********************************newly added**************************************/

  }

  for (int i = 0; i < (int)p.num_nodes; i++)
  {
    ptr_t current = p.first_node + i;
    double offset = fa_node_offset.read(current);
    double result;
    if (rt->safe_cast(ctx, current, pvt_region))
      result = fa_pvt_result.read(current);
    else
      result = fa_shr_result.read(current); 
    //printf("previous value is %f\n", result);
    //printf("current value is %f + %f = %f\n", result, offset, offset + result);
    fa_node_result.write(current, offset + result);
  }

  //printf("end of a task from piece %d\n", piece_num);
}






CheckTask::CheckTask(LogicalPartition lp,
                     LogicalRegion lr,
                     FieldID fid,
                     const Domain &launch_domain,
                     const ArgumentMap &arg_map)
 : IndexLauncher(CheckTask::TASK_ID, launch_domain, TaskArgument(), arg_map,
                 Predicate::TRUE_PRED, false/*must*/, CheckTask::MAPPER_ID)
{
  RegionRequirement rr_check(lp, 0/*identity*/, READ_ONLY, EXCLUSIVE, lr);
  rr_check.add_field(fid);
  add_region_requirement(rr_check);
}

/*static*/
const char * const CheckTask::TASK_NAME = "check_task";

bool CheckTask::dispatch(Context ctx, HighLevelRuntime *runtime, bool success)
{
  FutureMap fm = runtime->execute_index_space(ctx, *this);
  fm.wait_all_results();
  Rect<1> launch_array = launch_domain.get_rect<1>();
  for (GenericPointInRectIterator<1> pir(launch_array); pir; pir++)
    success = fm.get_result<bool>(DomainPoint::from_point<1>(pir.p)) && success;
  return success;
}

/*static*/
bool CheckTask::cpu_impl(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, HighLevelRuntime *runtime)
{
  RegionAccessor<AccessorType::Generic, float> fa_check = 
    regions[0].get_field_accessor(task->regions[0].instance_fields[0]).typeify<float>();
  LogicalRegion lr = task->regions[0].region;
  IndexIterator itr(runtime, ctx, lr);
  bool success = true;
  while (itr.has_next() && success)
  {
    ptr_t ptr = itr.next();
    float value = fa_check.read(ptr);
    if (isnan(value))
      success = false;
  }
  return success;
}

/*static*/
void CheckTask::register_task(void)
{
  HighLevelRuntime::register_legion_task<bool, cpu_impl>(CheckTask::TASK_ID, Processor::LOC_PROC,
                                                         false/*single*/, true/*index*/,
                                                         CIRCUIT_CPU_LEAF_VARIANT,
                                                         TaskConfigOptions(CheckTask::LEAF),
                                                         CheckTask::TASK_NAME);
}

