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


#include <cstdio>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <time.h>

#include "circuit.h"
#include "circuit_mapper.h"
#include "legion.h"

using namespace LegionRuntime::HighLevel;
using namespace LegionRuntime::Accessor;

void fill_mat(std::vector<std::vector<double> > &mat, std::vector<double> &vec, int size)
{
  for (int i = 0; i < size; i++)
    {
        vec[i] = drand48() * 20;
        for (int j = 0; j < size; j++)
          if (i <= j && drand48() * 100 < 30)
          {
            mat[i][j] = drand48() * 30;
            mat[j][i] = mat[i][j];
          }
     }
}

void dense_to_sparse(std::vector<std::vector<double> >&mat, std::vector<SparseElem>&sparse_mat) 
{
  int size = mat.size();
  for (int i = 0; i < size; i++)
    for (int j = i; j < size; j++)
      if (mat[i][j] > 1e-5)
      {
        SparseElem temp;
        temp.x = i;
        temp.y = j;
        temp.z = mat[i][j];
        sparse_mat.push_back(temp);
      }
}
                                                                                                                                  
void print_mat(std::vector<std::vector<double> > &mat, std::vector<SparseElem>&sparse_mat, std::vector<double> &vec)
{
  printf("The dense matrix is:\n");
  for (unsigned int i = 0; i < mat.size(); i++)
  {
    for (unsigned int j = 0; j < mat.size(); j++)
      printf("%f ", mat[i][j]);
      printf("\n");
  }
  printf("\n\n\nThe sparse matrix is:\n\n");

  for (unsigned int i = 0; i < sparse_mat.size(); i++)
    printf("%d, %d, %f\n", sparse_mat[i].x, sparse_mat[i].y, sparse_mat[i].z);
  printf("\n The vector is :\n");
  for (unsigned int i = 0; i < vec.size(); i++)
    printf("%f, ", vec[i]);
  printf("\n");
}

int calc_wire_num(std::vector<SparseElem> &sparse_mat)
{
  /*
  int num_wires = 0;
  for (unsigned int i = 0; i < sparse_mat.size(); i++)
    if (sparse_mat[i].x == sparse_mat[i].y) 
      num_wires += 2;
    else
      num_wires++;
  num_wires /= 2; 
  return num_wires;
  */
  return (int) sparse_mat.size();
}

LegionRuntime::Logger::Category log_circuit("circuit");

// Utility functions (forward declarations)
void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
                      int &steps, int &sync, bool &perform_checks, bool &dump_values, int &size);

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed, int steps,
                        std::vector<SparseElem>&sparse_mat, std::vector<double> &vec);

void allocate_node_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace node_space);
void allocate_wire_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace wire_space);
void allocate_locator_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace locator_space);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, HighLevelRuntime *runtime)
{
  int num_loops = 2;
  int num_pieces = 3;
  int nodes_per_piece = 2; // This one should be calculated instead
  int wires_per_piece = 4;
  int pct_wire_in_piece = 95;
  int random_seed = 12345;
  int steps = STEPS;
  int sync = 0;
  bool perform_checks = false;
  bool dump_values = false;
  int size = 7;
  
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;

    parse_input_args(argv, argc, num_loops, num_pieces, nodes_per_piece, 
		     wires_per_piece, pct_wire_in_piece, random_seed,
		     steps, sync, perform_checks, dump_values, size);
    std::vector<std::vector<double> > mat(size, std::vector<double>(size, 0)); // the size x size matrix is initially filled with 0s.
    std::vector<double> vec(size, 0); // This is the vector which is going to multiply the matrix.
    std::vector<SparseElem> sparse_mat;
    fill_mat(mat, vec, size);
    dense_to_sparse(mat, sparse_mat); 
    print_mat(mat, sparse_mat, vec);

    log_circuit.print("circuit settings: loops=%d pieces=%d nodes/piece=%d "
                            "wires/piece=%d pct_in_piece=%d seed=%d",
       num_loops, num_pieces, nodes_per_piece, wires_per_piece, pct_wire_in_piece, random_seed);
  

  Circuit circuit;
  {
    int num_circuit_nodes = size;
    int num_circuit_wires = (int)sparse_mat.size();// This may make the code not working well
    nodes_per_piece = (num_circuit_nodes % num_pieces == 0) ? (num_circuit_nodes
    / num_pieces) : (num_circuit_nodes / num_pieces + 1);
    //printf("the wire number is %d\n", calc_wire_num(sparse_mat));

    // Make index spaces
    IndexSpace node_index_space = runtime->create_index_space(ctx,num_circuit_nodes);
    runtime->attach_name(node_index_space, "node_index_space");
    IndexSpace wire_index_space = runtime->create_index_space(ctx,num_circuit_wires);
    runtime->attach_name(wire_index_space, "wire_index_space");
    // Make field spaces
    FieldSpace node_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(node_field_space, "node_field_space");
    FieldSpace wire_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(wire_field_space, "wire_field_space");
    FieldSpace locator_field_space = runtime->create_field_space(ctx);
    runtime->attach_name(locator_field_space, "locator_field_space");
    // Allocate fields
    allocate_node_fields(ctx, runtime, node_field_space);
    allocate_wire_fields(ctx, runtime, wire_field_space);
    allocate_locator_fields(ctx, runtime, locator_field_space);
    // Make logical regions
    circuit.all_nodes = runtime->create_logical_region(ctx,node_index_space,node_field_space);
    runtime->attach_name(circuit.all_nodes, "all_nodes");
    circuit.all_wires = runtime->create_logical_region(ctx,wire_index_space,wire_field_space);
    runtime->attach_name(circuit.all_wires, "all_wires");
    circuit.node_locator = runtime->create_logical_region(ctx,node_index_space,locator_field_space);
    runtime->attach_name(circuit.node_locator, "node_locator");
  }

  // Load the circuit
  std::vector<CircuitPiece> pieces(num_pieces);
  Partitions parts = load_circuit(circuit, pieces, ctx, runtime, num_pieces, nodes_per_piece,
                                  wires_per_piece, pct_wire_in_piece,
                                  random_seed, steps, sparse_mat, vec);

  // Arguments for each point
  ArgumentMap local_args;
  for (int idx = 0; idx < num_pieces; idx++)
  {
    DomainPoint point = DomainPoint::from_point<1>(Point<1>(idx));
    local_args.set_point(point, TaskArgument(&(pieces[idx]),sizeof(CircuitPiece)));
  }

  // Make the launchers
  Rect<1> launch_rect(Point<1>(0), Point<1>(num_pieces-1)); 
  Domain launch_domain = Domain::from_rect<1>(launch_rect);
  CalcNewCurrentsTask cnc_launcher(parts.pvt_wires, parts.pvt_nodes, parts.shr_nodes, parts.ghost_nodes,
                                   circuit.all_wires, circuit.all_nodes, launch_domain, local_args);

  printf("Starting main simulation loop\n");
  double ts_start, ts_end;
  ts_start = LegionRuntime::TimeStamp::get_current_time_in_micros();
  // Run the main loop
  bool simulation_success = true;

  //printf("Before the task!!!\n");
  TaskHelper::dispatch_task<CalcNewCurrentsTask>(cnc_launcher, ctx, runtime, 
                                                 perform_checks, simulation_success, true);
  
  //printf("After the task!!!\n");
  ts_end = LegionRuntime::TimeStamp::get_current_time_in_micros();

  printf("The result of the matrix multiplication is :\n");
  {
    RegionRequirement nodes_req(circuit.all_nodes, READ_ONLY, EXCLUSIVE, circuit.all_nodes);
    nodes_req.add_field(FID_NODE_RESULT);
    IndexIterator itr(runtime, ctx, circuit.all_nodes);
    PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
    RegionAccessor<AccessorType::Generic, double> fa_node_result =
        nodes.get_field_accessor(FID_NODE_RESULT).typeify<double>();
    while (itr.has_next())
    {
      ptr_t node_ptr = itr.next();
      printf("%f\n", fa_node_result.read(node_ptr));
    }
  }
  printf("The correct answer should be: \n");
  std::vector<double> result(size, 0);
  for (unsigned int i = 0; i < vec.size(); i++)
  {
    for (unsigned int j = 0; j < vec.size(); j++)
    {
      result[i] += mat[i][j] * vec[j];
    }
    printf("%f\n", result[i]);
  }
  if (simulation_success)
    printf("SUCCESS!\n");
  else
    printf("FAILURE!\n");
  {
    double sim_time = 1e-6 * (ts_end - ts_start);
    printf("ELAPSED TIME = %7.3f s\n", sim_time);

    // Compute the floating point operations per second
    long num_circuit_nodes = num_pieces * nodes_per_piece;
    long num_circuit_wires = num_pieces * wires_per_piece;
    // calculate currents
    long operations = num_circuit_wires * (WIRE_SEGMENTS*6 + (WIRE_SEGMENTS-1)*4) * steps;
    // distribute charge
    operations += (num_circuit_wires * 4);
    // update voltages
    operations += (num_circuit_nodes * 4);
    // multiply by the number of loops
    operations *= num_loops;

    // Compute the number of gflops
    double gflops = (1e-9*operations)/sim_time;
    printf("GFLOPS = %7.3f GFLOPS\n", gflops);
  }
  log_circuit.print("simulation complete - destroying regions");

  if (dump_values)
  {
    RegionRequirement wires_req(circuit.all_wires, READ_ONLY, EXCLUSIVE, circuit.all_wires);
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      wires_req.add_field(FID_CURRENT+i);
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      wires_req.add_field(FID_WIRE_VOLTAGE+i);
    PhysicalRegion wires = runtime->map_region(ctx, wires_req);
    wires.wait_until_valid();
    RegionAccessor<AccessorType::Generic, float> fa_wire_currents[WIRE_SEGMENTS];
    for (int i = 0; i < WIRE_SEGMENTS; i++)
      fa_wire_currents[i] = wires.get_field_accessor(FID_CURRENT+i).typeify<float>();
    RegionAccessor<AccessorType::Generic, float> fa_wire_voltages[WIRE_SEGMENTS-1];
    for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
      fa_wire_voltages[i] = wires.get_field_accessor(FID_WIRE_VOLTAGE+i).typeify<float>();
    IndexIterator itr(runtime, ctx, circuit.all_wires.get_index_space());
    while (itr.has_next())
    {
      ptr_t wire_ptr = itr.next();
      for (int i = 0; i < WIRE_SEGMENTS; ++i)
        printf(" %.5g", fa_wire_currents[i].read(wire_ptr));
      for (int i = 0; i < WIRE_SEGMENTS - 1; ++i)
        printf(" %.5g", fa_wire_voltages[i].read(wire_ptr));
      printf("\n");
    }
    runtime->unmap_region(ctx, wires);
  }

  // Now we can destroy all the things that we made
  {
    runtime->destroy_logical_region(ctx,circuit.all_nodes);
    runtime->destroy_logical_region(ctx,circuit.all_wires);
    runtime->destroy_logical_region(ctx,circuit.node_locator);
    runtime->destroy_field_space(ctx,circuit.all_nodes.get_field_space());
    runtime->destroy_field_space(ctx,circuit.all_wires.get_field_space());
    runtime->destroy_field_space(ctx,circuit.node_locator.get_field_space());
    runtime->destroy_index_space(ctx,circuit.all_nodes.get_index_space());
    runtime->destroy_index_space(ctx,circuit.all_wires.get_index_space());
  }
}

static void update_mappers(Machine machine, HighLevelRuntime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new CircuitMapper(machine, rt, *it), *it);
  }
}

int main(int argc, char **argv)
{
  HighLevelRuntime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  HighLevelRuntime::register_legion_task<top_level_task>(TOP_LEVEL_TASK_ID,
      Processor::LOC_PROC, true/*single*/, false/*index*/,
      AUTO_GENERATE_ID, TaskConfigOptions(), "top_level");
  // If we're running on the shared low-level then only register cpu tasks
#ifdef SHARED_LOWLEVEL
  TaskHelper::register_cpu_variants<CalcNewCurrentsTask>();
  TaskHelper::register_cpu_variants<DistributeChargeTask>();
  TaskHelper::register_cpu_variants<UpdateVoltagesTask>();
#else
  TaskHelper::register_hybrid_variants<CalcNewCurrentsTask>();
  TaskHelper::register_hybrid_variants<DistributeChargeTask>();
  TaskHelper::register_hybrid_variants<UpdateVoltagesTask>();
#endif
  CheckTask::register_task();
  HighLevelRuntime::register_reduction_op<AccumulateCharge>(REDUCE_ID);
  HighLevelRuntime::set_registration_callback(update_mappers);

  return HighLevelRuntime::start(argc, argv);
}

void parse_input_args(char **argv, int argc, int &num_loops, int &num_pieces,
                      int &nodes_per_piece, int &wires_per_piece,
                      int &pct_wire_in_piece, int &random_seed,
                      int &steps, int &sync, bool &perform_checks,
                      bool &dump_values, int &size)
{
  for (int i = 1; i < argc; i++) 
  {
    if (!strcmp(argv[i], "-l")) 
    {
      num_loops = atoi(argv[++i]);
      continue;
    }

    if (!strcmp(argv[i], "-i")) 
    {
      steps = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-p")) 
    {
      num_pieces = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-npp")) 
    {
      nodes_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-wpp")) 
    {
      wires_per_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-pct")) 
    {
      pct_wire_in_piece = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-size")) 
    {
      size = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-s")) 
    {
      random_seed = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-sync")) 
    {
      sync = atoi(argv[++i]);
      continue;
    }

    if(!strcmp(argv[i], "-checks"))
    {
      perform_checks = true;
      continue;
    }

    if(!strcmp(argv[i], "-dump"))
    {
      dump_values = true;
      continue;
    }
  }
}

void allocate_node_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace node_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, node_space);
  allocator.allocate_field(sizeof(float), FID_NODE_CAP);
  runtime->attach_name(node_space, FID_NODE_CAP, "node capacitance");
  allocator.allocate_field(sizeof(float), FID_LEAKAGE);
  runtime->attach_name(node_space, FID_LEAKAGE, "leakage");
  allocator.allocate_field(sizeof(float), FID_CHARGE);
  runtime->attach_name(node_space, FID_CHARGE, "charge");
  allocator.allocate_field(sizeof(float), FID_NODE_VOLTAGE);
  runtime->attach_name(node_space, FID_NODE_VOLTAGE, "node voltage");

  allocator.allocate_field(sizeof(double), FID_NODE_VALUE);
  runtime->attach_name(node_space, FID_NODE_VALUE, "node value");
  allocator.allocate_field(sizeof(double), FID_NODE_RESULT);
  runtime->attach_name(node_space, FID_NODE_RESULT, "node result");
}

void allocate_wire_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace wire_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, wire_space);
  allocator.allocate_field(sizeof(ptr_t), FID_IN_PTR);
  runtime->attach_name(wire_space, FID_IN_PTR, "in_ptr");
  allocator.allocate_field(sizeof(ptr_t), FID_OUT_PTR);
  runtime->attach_name(wire_space, FID_OUT_PTR, "out_ptr");
  allocator.allocate_field(sizeof(PointerLocation), FID_IN_LOC);
  runtime->attach_name(wire_space, FID_IN_LOC, "in_loc");
  allocator.allocate_field(sizeof(PointerLocation), FID_OUT_LOC);
  runtime->attach_name(wire_space, FID_OUT_LOC, "out_loc");
  allocator.allocate_field(sizeof(float), FID_INDUCTANCE);
  runtime->attach_name(wire_space, FID_INDUCTANCE, "inductance");
  allocator.allocate_field(sizeof(float), FID_RESISTANCE);
  runtime->attach_name(wire_space, FID_RESISTANCE, "resistance");
  allocator.allocate_field(sizeof(float), FID_WIRE_CAP);
  runtime->attach_name(wire_space, FID_WIRE_CAP, "wire capacitance");
  for (int i = 0; i < WIRE_SEGMENTS; i++)
  {
    char field_name[10];
    allocator.allocate_field(sizeof(float), FID_CURRENT+i);
    sprintf(field_name, "current_%d", i);
    runtime->attach_name(wire_space, FID_CURRENT+i, field_name);
  }
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
  {
    char field_name[15];
    allocator.allocate_field(sizeof(float), FID_WIRE_VOLTAGE+i);
    sprintf(field_name, "wire_voltage_%d", i);
    runtime->attach_name(wire_space, FID_WIRE_VOLTAGE+i, field_name);
  }

  allocator.allocate_field(sizeof(double), FID_WIRE_VALUE);
  runtime->attach_name(wire_space, FID_WIRE_VALUE, "wire value");
  allocator.allocate_field(sizeof(int), FID_PIECE_NUM1);
  runtime->attach_name(wire_space, FID_PIECE_NUM1, "piece num1");
  allocator.allocate_field(sizeof(int), FID_PIECE_NUM2);
  runtime->attach_name(wire_space, FID_PIECE_NUM2, "piece num2");
}

void allocate_locator_fields(Context ctx, HighLevelRuntime *runtime, FieldSpace locator_space)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, locator_space);
  allocator.allocate_field(sizeof(PointerLocation), FID_LOCATOR);
  runtime->attach_name(locator_space, FID_LOCATOR, "locator");
}

PointerLocation find_location(ptr_t ptr, const std::set<ptr_t> &private_nodes,
                              const std::set<ptr_t> &shared_nodes, const std::set<ptr_t> &ghost_nodes)
{
  if (private_nodes.find(ptr) != private_nodes.end())
  {
    return PRIVATE_PTR;
  }
  else if (shared_nodes.find(ptr) != shared_nodes.end())
  {
    return SHARED_PTR;
  }
  else if (ghost_nodes.find(ptr) != ghost_nodes.end())
  {
    return GHOST_PTR;
  }
  // Should never make it here, if we do something bad happened
  assert(false);
  return PRIVATE_PTR;
}

template<typename T>
static T random_element(const std::set<T> &set)
{
  int index = int(drand48() * set.size());
  typename std::set<T>::const_iterator it = set.begin();
  while (index-- > 0) it++;
  return *it;
}

template<typename T>
static T random_element(const std::vector<T> &vec)
{
  int index = int(drand48() * vec.size());
  return vec[index];
}

Partitions load_circuit(Circuit &ckt, std::vector<CircuitPiece> &pieces, Context ctx,
                        HighLevelRuntime *runtime, int num_pieces, int nodes_per_piece,
                        int wires_per_piece, int pct_wire_in_piece, int random_seed, int steps, std::vector<SparseElem>&sparse_mat, 
                        std::vector<double> &vec)
{
  log_circuit.print("Initializing matrix multiplication...");
  // inline map physical instances for the nodes and wire regions
  RegionRequirement wires_req(ckt.all_wires, READ_WRITE, EXCLUSIVE, ckt.all_wires);
  wires_req.add_field(FID_IN_PTR);
  wires_req.add_field(FID_OUT_PTR);
  wires_req.add_field(FID_IN_LOC);
  wires_req.add_field(FID_OUT_LOC);
  wires_req.add_field(FID_INDUCTANCE);
  wires_req.add_field(FID_RESISTANCE);
  wires_req.add_field(FID_WIRE_CAP);
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    wires_req.add_field(FID_CURRENT+i);
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    wires_req.add_field(FID_WIRE_VOLTAGE+i);
  wires_req.add_field(FID_WIRE_VALUE);
  wires_req.add_field(FID_PIECE_NUM1);
  wires_req.add_field(FID_PIECE_NUM2);

  RegionRequirement nodes_req(ckt.all_nodes, READ_WRITE, EXCLUSIVE, ckt.all_nodes);
  nodes_req.add_field(FID_NODE_CAP);
  nodes_req.add_field(FID_LEAKAGE);
  nodes_req.add_field(FID_CHARGE);
  nodes_req.add_field(FID_NODE_VOLTAGE);
  nodes_req.add_field(FID_NODE_VALUE);
  nodes_req.add_field(FID_NODE_RESULT);

  RegionRequirement locator_req(ckt.node_locator, READ_WRITE, EXCLUSIVE, ckt.node_locator);
  locator_req.add_field(FID_LOCATOR);
  PhysicalRegion wires = runtime->map_region(ctx, wires_req);
  PhysicalRegion nodes = runtime->map_region(ctx, nodes_req);
  PhysicalRegion locator = runtime->map_region(ctx, locator_req);


  Coloring wire_owner_map;
  Coloring private_node_map;
  Coloring shared_node_map;
  Coloring ghost_node_map;
  Coloring locator_node_map;

  Coloring privacy_map;
  privacy_map[0];
  privacy_map[1];

  // keep a O(1) indexable list of nodes in each piece for connecting wires
  std::vector<std::vector<ptr_t> > piece_node_ptrs(num_pieces);
  std::vector<int> piece_shared_nodes(num_pieces, 0); // num of shared nodes in each piece

  srand48(random_seed);

  nodes.wait_until_valid();
  RegionAccessor<AccessorType::Generic, float> fa_node_cap = 
    nodes.get_field_accessor(FID_NODE_CAP).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_node_leakage = 
    nodes.get_field_accessor(FID_LEAKAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_node_charge = 
    nodes.get_field_accessor(FID_CHARGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_node_voltage = 
    nodes.get_field_accessor(FID_NODE_VOLTAGE).typeify<float>();
  RegionAccessor<AccessorType::Generic, double> fa_node_value = 
    nodes.get_field_accessor(FID_NODE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, double> fa_node_result = 
    nodes.get_field_accessor(FID_NODE_RESULT).typeify<double>();

  locator.wait_until_valid();
  RegionAccessor<AccessorType::Generic, PointerLocation> locator_acc = 
    locator.get_field_accessor(FID_LOCATOR).typeify<PointerLocation>();

  int num_nodes = (int)vec.size();

  ptr_t *first_nodes = new ptr_t[num_pieces];
  {
    IndexAllocator node_allocator = runtime->create_index_allocator(ctx, ckt.all_nodes.get_index_space());
    node_allocator.alloc(num_nodes);
  }
  // Write the values of the nodes.
  {
    IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space());
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        int current = n * nodes_per_piece + i;
        if (current >= num_nodes)
          break;
        assert(itr.has_next());
        ptr_t node_ptr = itr.next();
        if (i == 0)
          first_nodes[n] = node_ptr;

          float capacitance = drand48() + 1.f;
          fa_node_cap.write(node_ptr, capacitance);
          float leakage = 0.1f * drand48();
          fa_node_leakage.write(node_ptr, leakage);
          fa_node_charge.write(node_ptr, 0.f);
          float init_voltage = 2*drand48() - 1.f;
          fa_node_voltage.write(node_ptr, init_voltage);

        fa_node_value.write(node_ptr, vec[current]);
        fa_node_result.write(node_ptr, 0.0);

        // Just put everything in everyones private map at the moment       
        // We'll pull pointers out of here later as nodes get tied to 
        // wires that are non-local
        private_node_map[n].points.insert(node_ptr);
        privacy_map[0].points.insert(node_ptr);
        locator_node_map[n].points.insert(node_ptr);
	      piece_node_ptrs[n].push_back(node_ptr);
      }
    }
  }
  // verify the previous implementation
  IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space());
    for (int n = 0; n < num_pieces; n++)
    {
      printf("node values for the %d th piece:\n", n);
      for (int i = 0; i < nodes_per_piece; i++)
      {
        int current = n * nodes_per_piece + i;
        if (current >= num_nodes)
          break;
        assert(itr.has_next());
        ptr_t node_ptr = itr.next();
        printf("%f ", fa_node_value.read(node_ptr));
      }
      printf("There are %d nodes in this piece\n", (int)piece_node_ptrs[n].size());
      printf("\n");
    }
  printf("\n");

  wires.wait_until_valid();
  RegionAccessor<AccessorType::Generic, float> fa_wire_currents[WIRE_SEGMENTS];
  for (int i = 0; i < WIRE_SEGMENTS; i++)
    fa_wire_currents[i] = wires.get_field_accessor(FID_CURRENT+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_wire_voltages[WIRE_SEGMENTS-1];
  for (int i = 0; i < (WIRE_SEGMENTS-1); i++)
    fa_wire_voltages[i] = wires.get_field_accessor(FID_WIRE_VOLTAGE+i).typeify<float>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_wire_in_ptr = 
    wires.get_field_accessor(FID_IN_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, ptr_t> fa_wire_out_ptr = 
    wires.get_field_accessor(FID_OUT_PTR).typeify<ptr_t>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_wire_in_loc = 
    wires.get_field_accessor(FID_IN_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, PointerLocation> fa_wire_out_loc = 
    wires.get_field_accessor(FID_OUT_LOC).typeify<PointerLocation>();
  RegionAccessor<AccessorType::Generic, float> fa_wire_inductance = 
    wires.get_field_accessor(FID_INDUCTANCE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_wire_resistance = 
    wires.get_field_accessor(FID_RESISTANCE).typeify<float>();
  RegionAccessor<AccessorType::Generic, float> fa_wire_cap = 
    wires.get_field_accessor(FID_WIRE_CAP).typeify<float>();

  RegionAccessor<AccessorType::Generic, double> fa_wire_value = 
    wires.get_field_accessor(FID_WIRE_VALUE).typeify<double>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num1 = 
    wires.get_field_accessor(FID_PIECE_NUM1).typeify<int>();
  RegionAccessor<AccessorType::Generic, int> fa_piece_num2 = 
    wires.get_field_accessor(FID_PIECE_NUM2).typeify<int>();

  ptr_t *first_wires = new ptr_t[num_pieces];
  // Allocate all the wires
  int num_wires = (int)sparse_mat.size(); 
  {
    IndexAllocator wire_allocator = runtime->create_index_allocator(ctx, ckt.all_wires.get_index_space());
    wire_allocator.alloc(num_wires);
  }

  {
    IndexIterator itr(runtime, ctx, ckt.all_wires.get_index_space());
    for (int i = 0; i < num_wires; i++)
    {
      assert(itr.has_next());
      ptr_t wire_ptr = itr.next();
      // Record the first wire pointer for this piece

      for (int j = 0; j < WIRE_SEGMENTS; j++)
        fa_wire_currents[j].write(wire_ptr, 0.f);
      for (int j = 0; j < WIRE_SEGMENTS-1; j++) 
        fa_wire_voltages[j].write(wire_ptr, 0.f);

      float resistance = drand48() * 10.0 + 1.0;
      fa_wire_resistance.write(wire_ptr, resistance);
      // Keep inductance on the order of 1e-3 * dt to avoid resonance problems
      float inductance = (drand48() + 0.1) * DELTAT * 1e-3;
      fa_wire_inductance.write(wire_ptr, inductance);
      float capacitance = drand48() * 0.1;
      fa_wire_cap.write(wire_ptr, capacitance);
      

      /******************newly added****************/

      int m1 = sparse_mat[i].x / nodes_per_piece;
      int n1 = sparse_mat[i].x % nodes_per_piece;
      ptr_t p1 = piece_node_ptrs[m1][n1];
      fa_wire_in_ptr.write(wire_ptr, p1);


      int m2 = sparse_mat[i].y / nodes_per_piece;
      int n2 = sparse_mat[i].y % nodes_per_piece;
      ptr_t p2 = piece_node_ptrs[m2][n2];
      fa_wire_out_ptr.write(wire_ptr, p2);

      fa_wire_value.write(wire_ptr, sparse_mat[i].z); 

      fa_piece_num1.write(wire_ptr, m1); // corresponding to in_ptr
      fa_piece_num2.write(wire_ptr, m2); // corresponding to out_ptr
      // These nodes are no longer private
      if (m1 != m2) // If the two nodes are in different pieces
      {
        privacy_map[0].points.erase(p1);
        privacy_map[0].points.erase(p2);
        privacy_map[1].points.insert(p1);
        privacy_map[1].points.insert(p2);
        ghost_node_map[m1].points.insert(p2);
        ghost_node_map[m2].points.insert(p1);
        wire_owner_map[m1].points.insert(wire_ptr);
        wire_owner_map[m2].points.insert(wire_ptr);
      }
      else
        wire_owner_map[m1].points.insert(wire_ptr);

      /************newly added**********************/



      // balance the number of wires in pieces
      //if (wire_owner_map[m1].points.size() < wire_owner_map[m2].points.size())
      //  wire_owner_map[m1].points.insert(wire_ptr);
      //else
      //  wire_owner_map[m2].points.insert(wire_ptr);

    }
  }

  // Second pass: make some random fraction of the private nodes shared
  {
    IndexIterator itr(runtime, ctx, ckt.all_nodes.get_index_space()); 
    for (int n = 0; n < num_pieces; n++)
    {
      for (int i = 0; i < nodes_per_piece; i++)
      {
        int current = n * nodes_per_piece + i;
        if (current >= num_nodes)
          break;
        assert(itr.has_next());
        ptr_t node_ptr = itr.next();
        if (privacy_map[0].points.find(node_ptr) == privacy_map[0].points.end()) // if shared
        {
          private_node_map[n].points.erase(node_ptr);
          // node is now shared
          shared_node_map[n].points.insert(node_ptr);
          locator_acc.write(node_ptr,SHARED_PTR); // node is shared 
        }
        else
        {
          locator_acc.write(node_ptr,PRIVATE_PTR); // node is private 
        }
      }
    }
  }
  // Second pass (part 2): go through the wires and update the locations // ////////////////////////////// This part is useless
  {
    IndexIterator itr(runtime, ctx, ckt.all_wires.get_index_space());
    for (int i = 0; i < num_wires; i++)
    {
      assert(itr.has_next());
      ptr_t wire_ptr = itr.next();
      ptr_t in_ptr = fa_wire_in_ptr.read(wire_ptr);
      ptr_t out_ptr = fa_wire_out_ptr.read(wire_ptr);

      // Find out which piece does the wire belong to.
      int piece_num = 0;
      for (int m = 0; m < (int)wire_owner_map.size(); m++)
        if (wire_owner_map[m].points.find(wire_ptr) != wire_owner_map[m].points.end())
        {
          piece_num = m;
          break;
        }
     // printf("piece_num = %d\n\n", piece_num);      
      fa_wire_in_loc.write(wire_ptr, 
          find_location(in_ptr, private_node_map[piece_num].points, 
            shared_node_map[piece_num].points, ghost_node_map[piece_num].points));     
      fa_wire_out_loc.write(wire_ptr, 
          find_location(out_ptr, private_node_map[piece_num].points, 
            shared_node_map[piece_num].points, ghost_node_map[piece_num].points));
    }
  }

  runtime->unmap_region(ctx, wires);
  runtime->unmap_region(ctx, nodes);
  runtime->unmap_region(ctx, locator);

  // Now we can create our partitions and update the circuit pieces

  // first create the privacy partition that splits all the nodes into either shared or private
  IndexPartition privacy_part = runtime->create_index_partition(ctx, ckt.all_nodes.get_index_space(), privacy_map, true/*disjoint*/);
  runtime->attach_name(privacy_part, "is_private");
  
  IndexSpace all_private = runtime->get_index_subspace(ctx, privacy_part, 0);
  runtime->attach_name(all_private, "private");
  IndexSpace all_shared  = runtime->get_index_subspace(ctx, privacy_part, 1);
  runtime->attach_name(all_shared, "shared");

  // Now create partitions for each of the subregions
  Partitions result;
  IndexPartition priv = runtime->create_index_partition(ctx, all_private, private_node_map, true/*disjoint*/);
  runtime->attach_name(priv, "private");
  result.pvt_nodes = runtime->get_logical_partition_by_tree(ctx, priv, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.pvt_nodes, "private_nodes");
  IndexPartition shared = runtime->create_index_partition(ctx, all_shared, shared_node_map, true/*disjoint*/);
  runtime->attach_name(shared, "shared");
  result.shr_nodes = runtime->get_logical_partition_by_tree(ctx, shared, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.shr_nodes, "shared_nodes");
  IndexPartition ghost = runtime->create_index_partition(ctx, all_shared, ghost_node_map, false/*disjoint*/);
  runtime->attach_name(ghost, "ghost");
  result.ghost_nodes = runtime->get_logical_partition_by_tree(ctx, ghost, ckt.all_nodes.get_field_space(), ckt.all_nodes.get_tree_id());
  runtime->attach_name(result.ghost_nodes, "ghost_nodes");

  IndexPartition pvt_wires = runtime->create_index_partition(ctx, ckt.all_wires.get_index_space(), wire_owner_map, false/*disjoint*/);
  runtime->attach_name(pvt_wires, "private");
  result.pvt_wires = runtime->get_logical_partition_by_tree(ctx, pvt_wires, ckt.all_wires.get_field_space(), ckt.all_wires.get_tree_id()); 
  runtime->attach_name(result.pvt_wires, "private_wires");

  IndexPartition locs = runtime->create_index_partition(ctx, ckt.node_locator.get_index_space(), locator_node_map, true/*disjoint*/);
  runtime->attach_name(locs, "locs");
  result.node_locations = runtime->get_logical_partition_by_tree(ctx, locs, ckt.node_locator.get_field_space(), ckt.node_locator.get_tree_id());
  runtime->attach_name(result.node_locations, "node_locations");

  char buf[100];
  // Build the pieces
  for (int n = 0; n < num_pieces; n++)
  {
    pieces[n].pvt_nodes = runtime->get_logical_subregion_by_color(ctx, result.pvt_nodes, n);
    sprintf(buf, "private_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_nodes, buf);
    pieces[n].shr_nodes = runtime->get_logical_subregion_by_color(ctx, result.shr_nodes, n);
    sprintf(buf, "shared_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].shr_nodes, buf);
    pieces[n].ghost_nodes = runtime->get_logical_subregion_by_color(ctx, result.ghost_nodes, n);
    sprintf(buf, "ghost_nodes_of_piece_%d", n);
    runtime->attach_name(pieces[n].ghost_nodes, buf);
    pieces[n].pvt_wires = runtime->get_logical_subregion_by_color(ctx, result.pvt_wires, n);
    sprintf(buf, "private_wires_of_piece_%d", n);
    runtime->attach_name(pieces[n].pvt_wires, buf);

    pieces[n].num_wires = wire_owner_map[n].points.size();
    pieces[n].first_wire = first_wires[n];
    pieces[n].num_nodes = piece_node_ptrs[n].size();
    pieces[n].first_node = first_nodes[n];

    pieces[n].dt = DELTAT;
    pieces[n].steps = steps;
    pieces[n].piece_num = n;
  }

  delete [] first_wires;
  delete [] first_nodes;

  log_circuit.print("Finished initializing simulation...");

  return result;
}

