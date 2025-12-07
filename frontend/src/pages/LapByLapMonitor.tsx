export default function LapByLapMonitor() {
  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold">Lap-by-Lap Monitor</h1>
      
      <div className="card">
        <p className="text-gray-400 text-center py-20">
          Lap time and degradation charts will be displayed here.
          <br />
          Charts include: Lap time progression, tire degradation curves, stint summary table.
        </p>
      </div>
    </div>
  );
}
