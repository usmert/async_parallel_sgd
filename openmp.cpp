#include "common.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <omp.h>
#include <unordered_set>
// Apply the force from neighbor to particle

static struct SimulationParams {
 double cellSize;
 int gridSize;
    std::vector<std::vector<std::unordered_set<particle_t*>>> grid;
    std::vector<std::vector<omp_lock_t>> gridLocks;
} simParams;

void apply_force(particle_t& particle, particle_t& neighbor) {
    // Calculate Distance
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    // Check if the two particles should interact
    if (r2 > cutoff * cutoff)
        return;

    r2 = fmax(r2, min_r * min_r);
    double r = sqrt(r2);

    // Very simple short-range repulsive force
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

// Integrate the ODE
void move(particle_t& p, double size) {
    // Slightly simplified Velocity Verlet integration
    // Conserves energy better than explicit Euler method

    int startblockX = static_cast<int>(p.x / simParams.cellSize);
    int startblockY = static_cast<int>(p.y / simParams.cellSize);


    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Bounce from walls
    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }

    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }

    int endblockX = static_cast<int>(p.x / simParams.cellSize);
    int endblockY = static_cast<int>(p.y / simParams.cellSize);


    if (endblockX != startblockX || startblockY != endblockY) {
        omp_set_lock(&simParams.gridLocks[endblockX][endblockY]);
        simParams.grid[endblockX][endblockY].insert(&p);
        omp_unset_lock(&simParams.gridLocks[endblockX][endblockY]);

        omp_set_lock(&simParams.gridLocks[startblockX][startblockY]);
        simParams.grid[startblockX][startblockY].erase(&p);
        omp_unset_lock(&simParams.gridLocks[startblockX][startblockY]);
    }

}

void init_simulation(particle_t* parts, int num_parts, double size) {
 // You can use this space to initialize static, global data objects
    // that you may need. This function will be called once before the
    // algorithm begins. Do not do any particle simulation here
 simParams.cellSize = fmax(size / num_parts, cutoff);
    simParams.gridSize = ceil(size / simParams.cellSize); 

    simParams.grid.resize(simParams.gridSize);
    simParams.gridLocks.resize(simParams.gridSize);
    for (int i = 0; i < simParams.gridSize; ++i) {
        simParams.grid[i].resize(simParams.gridSize);
        simParams.gridLocks[i].resize(simParams.gridSize);
        for (int j = 0; j < simParams.gridSize; ++j) {
            simParams.grid[i][j] = std::unordered_set<particle_t*>();
            omp_init_lock(&simParams.gridLocks[i][j]);
        }
    }
    
    for (int i = 0; i < num_parts; ++i) {
        int gridX = static_cast<int>(parts[i].x / simParams.cellSize);
        int gridY = static_cast<int>(parts[i].y / simParams.cellSize);
        simParams.grid[gridX][gridY].insert(&parts[i]);
    }
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    #pragma omp parallel for
    for (int gridX = 0; gridX < simParams.gridSize; ++gridX) { 
        #pragma omp parallel for
        for (int gridY = 0; gridY < simParams.gridSize; ++gridY) {
            int startX = std::max(0, gridX - 1);
            int startY = std::max(0, gridY - 1);
            int endX = std::min((int)simParams.gridSize - 1, gridX + 1);
            int endY = std::min((int)simParams.gridSize - 1, gridY + 1);
            for (particle_t* p: simParams.grid[gridX][gridY]) {
                p->ax = 0;
                p->ay = 0;
                for (int x = startX; x <= endX; ++x) {
                    for (int y = startY; y <= endY; ++y) {
                        for (particle_t* n : simParams.grid[x][y]) {
                                apply_force(*p, *n);
                        }
                    }
                }
            }
        }
    }

    #pragma omp parallel for 
    for (int i = 0; i < num_parts; ++i) {
        move(parts[i], size);
    }
}